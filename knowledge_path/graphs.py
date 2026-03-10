from neo4j import GraphDatabase
import csv
import sys
import json
import os
from collections import defaultdict

# Import base directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths_config import DATA_DIR


# Increase field size limit bcz of large csv fields
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


class Neo4jLoader:
    def __init__(self, uri, user, password, database="neo4jv8"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
    
    def close(self):
        self.driver.close()
    
    def create_constraints(self):
        """Create uniqueness constraint on CUI to improve performance"""
        with self.driver.session(database=self.database) as session:
            try:
                session.run("CREATE CONSTRAINT concept_cui IF NOT EXISTS FOR (c:Concept) REQUIRE c.cui IS UNIQUE")
            except Exception as e:
                print(f"Constraint may already exist: {e}")
    
    def load_nodes(self, csv_file, batch_size=1000000):
        """Load nodes from CSV file in batches"""
        
        with self.driver.session(database=self.database) as session:
            batch = []
            count = 0
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Convert header format: 'cui:ID' -> 'cui', ':LABEL' -> 'LABEL'
                    cui = row.get('cui:ID')
                    name = row.get('name')
                    label = row.get(':LABEL', 'Concept')
                    
                    if cui:
                        batch.append({'cui': cui, 'name': name})
                        count += 1
                        
                        if len(batch) >= batch_size:
                            self._create_nodes_batch(session, batch)
                            print(f"  Loaded {count} nodes...")
                            batch = []
                
                # Load remaining nodes
                if batch:
                    self._create_nodes_batch(session, batch)
                
    def _create_nodes_batch(self, session, batch):
        """Create a batch of nodes using UNWIND"""
        session.run("""
            UNWIND $batch AS node
            MERGE (c:Concept {cui: node.cui})
            SET c.name = node.name
        """, batch=batch)
    
    def load_relationships(self, csv_file, batch_size=1000000):
        """Load relationships from CSV file in batches"""        
        with self.driver.session(database=self.database) as session:
            batch = []
            count = 0
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Convert header format: ':START_ID' -> start_id, ':END_ID' -> end_id, ':TYPE' -> rel_type
                    start_id = row.get(':START_ID')
                    end_id = row.get(':END_ID')
                    rel_type = row.get(':TYPE', 'RELATED_TO')
                    rela = row.get('RELA', '')
                    
                    if start_id and end_id:
                        batch.append({'start_id': start_id, 'end_id': end_id,
                            'rel_type': rel_type, 'rela': rela})
                        count += 1
                        
                        if len(batch) >= batch_size:
                            self._create_relationships_batch(session, batch)
                            print(f"  Loaded {count} relationships...")
                            batch = []
                
                # Load remaining relationships
                if batch:
                    self._create_relationships_batch(session, batch)
                
    def _create_relationships_batch(self, session, batch):
        """Create a batch of relationships using UNWIND (without APOC)"""
        # Group relationships by type to create them with proper relationship types
        by_type = defaultdict(list)
        for rel in batch:
            # Use rela if available, otherwise use rel_type, or default to RELATED_TO
            # rel_key = rel['rela'] if rel['rela'] else (rel.get('rel_type') or 'RELATED_TO')
            # Use rela if available, otherwise 'RELATED_TO'
            rel_key = rel['rela'] if rel['rela'] else (rel.get('rel_type') or 'RELATED_TO')
            by_type[rel_key].append(rel)
        
        # Create relationships for each type separately
        for rel_type, rels in by_type.items():
            # Sanitize relationship type to ensure it's a valid Cypher identifier
            safe_rel_type = rel_type.replace('-', '_').replace(' ', '_').replace('/', '_')
            # Skip if empty after sanitization
            if not safe_rel_type:
                safe_rel_type = 'RELATED_TO'
            session.run(f"""
                UNWIND $batch AS rel
                MATCH (a:Concept {{cui: rel.start_id}})
                MATCH (b:Concept {{cui: rel.end_id}})
                CREATE (a)-[r:{safe_rel_type} {{rela: rel.rela}}]->(b)
            """, batch=rels)
            
    def update_node_definitions(self, csv_file, batch_size=1000):
        """Update existing nodes with definitions from CSV file"""
        with self.driver.session(database=self.database) as session:
            batch = []
            count = 0
            updated = 0
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    cui = row.get('cui:ID')
                    definition = row.get('definition', '')
                    
                    if cui and definition:
                        batch.append({'cui': cui, 'definition': definition})
                        count += 1
                        
                        if len(batch) >= batch_size:
                            result = self._update_definitions_batch(session, batch)
                            updated += result
                            print(f"  Processed {count} definitions, updated {updated} nodes...")
                            batch = []
                
                # Update remaining definitions
                if batch:
                    result = self._update_definitions_batch(session, batch)
                    updated += result
                    
            print(f"Total processed: {count} definitions, updated: {updated} nodes")
    
    def _update_definitions_batch(self, session, batch):
        """Update a batch of nodes with definitions"""
        result = session.run("""
            UNWIND $batch AS item
            MATCH (c:Concept {cui: item.cui})
            SET c.definition = item.definition
            RETURN count(c) as updated
        """, batch=batch)
        return result.single()['updated']


class Neo4jQuery:
    def __init__(self, uri, user, password, database="neo4jv8"):
        from neo4j import NotificationDisabledCategory
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(user, password),
            notifications_disabled_categories=[NotificationDisabledCategory.DEPRECATION]
        )
        self.database = database
    
    def close(self):
        self.driver.close()
    
    def search_entity(self, search_term):
        """Search for entities by name"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($search_term)
                RETURN c.cui as cui, c.name as name
                LIMIT 20
            """, search_term=search_term)
            
            entities = [dict(record) for record in result]
            return entities
    
    def get_entity_by_cui(self, cui):
        """Get entity by CUI"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (c:Concept {cui: $cui})
                RETURN c.cui as cui, c.name as name, c.tui as tui, c.tui_name as semantic_type
            """, cui=cui)
            
            entity = result.single()
            return dict(entity) if entity else None
    
    def get_entity_relationships(self, cui, limit=50):
        """Get all relationships for an entity"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (c:Concept {cui: $cui})-[r]-(related:Concept)
                RETURN c.cui as source_cui, c.name as source_name,
                       r.rela as rela,
                       related.cui as target_cui, related.name as target_name
                LIMIT $limit
            """, cui=cui, limit=limit)
            
            relationships = [dict(record) for record in result]
            return relationships
    
    def get_statistics(self):
        """Get database statistics"""
        with self.driver.session(database=self.database) as session:
            # Count nodes
            nodes_result = session.run("MATCH (c:Concept) RETURN count(c) as count")
            nodes_count = nodes_result.single()["count"]
            
            # Count relationships
            rels_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rels_count = rels_result.single()["count"]
            
            # Get relationship types distribution (sample for speed)
            rel_types_result = session.run("""
                MATCH ()-[r]->()
                WITH r.type as rel_type
                LIMIT 100000
                RETURN rel_type, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            rel_types = [dict(record) for record in rel_types_result]
            
            return {
                "total_concepts": nodes_count,
                "total_relationships": rels_count,
                "top_relationship_types": rel_types
            }
      
    def find_shortest_path(self, source_cui, target_cui, max_depth=5):
        """Find shortest path between two entities"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH path = shortestPath(
                    (source:Concept {cui: $source_cui})-[*..%s]-(target:Concept {cui: $target_cui})
                )
                WITH path, 
                     [n in nodes(path) | {cui: n.cui, name: n.name}] as node_list,
                     [r in relationships(path) | {type: type(r), rela: r.rela}] as rel_list
                RETURN node_list, rel_list, length(path) as path_length
            """ % max_depth, source_cui=source_cui, target_cui=target_cui)
            
            path_data = result.single()
            if path_data:
                return {
                    "nodes": path_data["node_list"],
                    "relationships": path_data["rel_list"],
                    "length": path_data["path_length"]
                }
            return None
    
    def find_all_paths(self, source_cui, target_cui, max_depth=4, limit=10):
        """Find all paths between two entities (limited)"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH path = (source:Concept {cui: $source_cui})-[*..%s]-(target:Concept {cui: $target_cui})
                WITH path, 
                     [n in nodes(path) | {cui: n.cui, name: n.name}] as node_list,
                     [r in relationships(path) | {type: type(r), rela: r.rela}] as rel_list
                RETURN node_list, rel_list, length(path) as path_length
                ORDER BY path_length
                LIMIT $limit
            """ % max_depth, source_cui=source_cui, target_cui=target_cui, limit=limit)
            
            paths = []
            for record in result:
                paths.append({
                    "nodes": record["node_list"],
                    "relationships": record["rel_list"],
                    "length": record["path_length"]
                })
            return paths
    
    def get_neighbors(self, cui, rel_type=None, direction="both", limit=20):
        """Get neighbors of a concept with optional filtering"""
        direction_clause = {
            "outgoing": "-[r]->",
            "incoming": "<-[r]-",
            "both": "-[r]-"
        }.get(direction, "-[r]-")
        
        type_filter = f"AND type(r) = '{rel_type}'" if rel_type else ""
        
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                MATCH (c:Concept {{cui: $cui}}){direction_clause}(neighbor:Concept)
                WHERE c <> neighbor {type_filter}
                RETURN neighbor.cui as cui, neighbor.name as name, 
                       r.rela as rela
                LIMIT $limit
            """, cui=cui, limit=limit)
            
            neighbors = [dict(record) for record in result]
            return neighbors
    
    def get_neighbors_filtered(
        self,
        cui,
        allowed_rels=None,
        direction="both",
        limit=15
    ):
        """
        Get meaningful filtered neighbors for grounding.
        """

        if allowed_rels is None:
            allowed_rels = [
                "has_associated_morphology",
                "has_finding_site",
                "has_grade",
                "has_stage",
                "has_biomarker",
                "part_of"
            ]

        direction_clause = {
            "outgoing": "-[r]->",
            "incoming": "<-[r]-",
            "both": "-[r]-"
        }.get(direction, "-[r]-")

        query = f"""
        MATCH (c:Concept {{cui: $cui}}){direction_clause}(n:Concept)
        WHERE c <> n
        AND r.rela IN $allowed_rels

        WITH DISTINCT n, r,
            CASE r.rela
            WHEN 'has_associated_morphology' THEN 1
            WHEN 'has_finding_site' THEN 2
            WHEN 'has_grade' THEN 3
            WHEN 'has_stage' THEN 4
            WHEN 'has_biomarker' THEN 5
            ELSE 10
            END AS priority

        RETURN
            n.cui   AS cui,
            n.name  AS name,
            r.rela  AS rela,
            priority

        ORDER BY priority ASC, n.name ASC
        LIMIT $limit
        """

        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                cui=cui,
                allowed_rels=allowed_rels,
                limit=limit
            )

            # Get CUI data into a dictionary
            cui_data = [dict(r) for r in result]

            # Remove duplicates based on cui
            unique_cui_data = []
            seen_cuis = set()
            for item in cui_data:
                if item['cui'] not in seen_cuis:
                    unique_cui_data.append(item)
                    seen_cuis.add(item['cui'])
            
            return unique_cui_data
   
    def get_subgraph(self, cui_list):
        """Get subgraph connecting a list of concepts"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (source:Concept)-[r]->(target:Concept)
                WHERE source.cui IN $cui_list AND target.cui IN $cui_list
                RETURN source.cui as source_cui, source.name as source_name,
                       r.rela as rela,
                       target.cui as target_cui, target.name as target_name
            """, cui_list=cui_list)
            
            edges = [dict(record) for record in result]
            return edges
    
    def get_concept_hierarchy(self, cui, depth=2, direction="children"):
        """Get hierarchical relationships (parent/child)"""
        if direction == "parents":
            rel_pattern = "<-[r:CHD|PAR|isa]-"
        else:  # children
            rel_pattern = "-[r:CHD|PAR|isa]->"
        
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                MATCH path = (c:Concept {{cui: $cui}}){rel_pattern}(related:Concept)
                WHERE length(path) <= $depth
                RETURN related.cui as cui, related.name as name,
                       type(r) as rel_type, length(path) as level
                ORDER BY level, related.name
                LIMIT 100
            """, cui=cui, depth=depth)
            
            hierarchy = [dict(record) for record in result]
            return hierarchy




if __name__ == '__main__':
    NEO4J_URL = "url" # Change this to your Neo4j URL
    NEO4J_USER = "user" # Change this to your Neo4j username
    NEO4J_PASSWORD = "password" # Change this to your Neo4j password
    
    NODES_CSV = os.path.join(DATA_DIR, "UMLS_sql", "neo4jdb", "v8", "nodes.csv")
    RELATIONSHIPS_CSV = os.path.join(DATA_DIR, "UMLS_sql", "neo4jdb", "v8", "relationships.csv")
    

    loader = Neo4jLoader(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, database="neo4jv8")  
    loader.create_constraints()
    # 
    loader.load_nodes(NODES_CSV, batch_size=100000)
# 
    loader.load_relationships(RELATIONSHIPS_CSV, batch_size=100000)

    loader.update_node_definitions(os.path.join(DATA_DIR, "UMLS_sql", "neo4jdb", "v8", "definitions.csv"), batch_size=1000)
    loader.close()



