import os
from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

PROJECT_ID = "creativo-bf5c8"
REGION = "us-central1"

aiplatform.init(project=PROJECT_ID, location=REGION)

# Use the resource names printed in your terminal
INDEX_RESOURCE_NAME = "projects/8368812350/locations/us-central1/indexes/7924903780032708608"
ENDPOINT_RESOURCE_NAME = "projects/8368812350/locations/us-central1/indexEndpoints/939926311097335808"

index = aiplatform.MatchingEngineIndex(INDEX_RESOURCE_NAME)
endpoint = aiplatform.MatchingEngineIndexEndpoint(ENDPOINT_RESOURCE_NAME)

print("Deploying index to endpoint (10-15 minutes)...")
endpoint.deploy_index(
    index=index,
    deployed_index_id="creativo_docs_deployed",
    display_name="creativo-docs-deployed",
    min_replica_count=1,
    max_replica_count=1,
)
print("✅ Index deployed!")

print("\n--- Save these in your .env ---")
print(f"VERTEX_INDEX_ID={INDEX_RESOURCE_NAME}")
print(f"VERTEX_INDEX_ENDPOINT_ID={ENDPOINT_RESOURCE_NAME}")
print(f"VERTEX_DEPLOYED_INDEX_ID=creativo_docs_deployed")