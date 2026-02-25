import os
from google.cloud import aiplatform
from dotenv import load_dotenv


load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

PROJECT_ID = "creativo-bf5c8"
REGION = "us-central1"
DISPLAY_NAME = "creativo-docs-index"
VECTOR_DIMENSION = 768  

aiplatform.init(project=PROJECT_ID, location=REGION)

# ---- Step 1: Create the Index ----
print("Creating index (this may take 20-30 minutes)...")
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name=DISPLAY_NAME,
    dimensions=VECTOR_DIMENSION,
    approximate_neighbors_count=10,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    index_update_method="STREAM_UPDATE",  # Allows real-time upserts
)
print(f"✅ Index created: {index.resource_name}")

# ---- Step 2: Create the Index Endpoint ----
print("\nCreating index endpoint...")
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name=f"{DISPLAY_NAME}-endpoint",
    public_endpoint_enabled=True,
)
print(f"✅ Endpoint created: {endpoint.resource_name}")

# ---- Step 3: Deploy the Index to the Endpoint ----
print("\nDeploying index to endpoint (this may take 10-15 minutes)...")
endpoint.deploy_index(
    index=index,
    deployed_index_id="creativo_docs_deployed",
    display_name="creativo-docs-deployed",
    min_replica_count=1,
    max_replica_count=1,
)
print("✅ Index deployed!")

# ---- Print the IDs you need ----
print("\n--- Save these values in your .env ---")
print(f"VERTEX_INDEX_ID={index.resource_name}")
print(f"VERTEX_INDEX_ENDPOINT_ID={endpoint.resource_name}")
print(f"VERTEX_DEPLOYED_INDEX_ID=creativo_docs_deployed")