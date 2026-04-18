import weaviate
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)

def test_weaviate_v4():
    print("Testing Weaviate v4 connection...")
    try:
        # Try default connection (v4)
        # Assuming 443 for REST and 50051 for gRPC
        # We'll try common variations
        
        host = os.getenv("WEAVIATE_HOST", "localhost")
        h_port = int(os.getenv("WEAVIATE_HTTP_PORT", 8080))
        h_secure = os.getenv("WEAVIATE_HTTP_SECURE", "False").lower() == "true"
        g_port = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
        g_secure = os.getenv("WEAVIATE_GRPC_SECURE", "False").lower() == "true"
        
        configs = [
            {"name": "Env Config", "http_port": h_port, "http_secure": h_secure, "grpc_port": g_port, "grpc_secure": g_secure},
        ]
        
        for config in configs:
            print(f"\n--- Trying Config: {config['name']} ({host}:{config['http_port']}) ---")
            client = weaviate.WeaviateClient(
                connection_params=weaviate.connect.ConnectionParams.from_params(
                    http_host=host,
                    http_port=config["http_port"],
                    http_secure=config["http_secure"],
                    grpc_host=host,
                    grpc_port=config["grpc_port"],
                    grpc_secure=config["grpc_secure"],
                )
            )
            try:
                client.connect()
                if client.is_ready():
                    print(f"SUCCESS: Weaviate is ready using {config['name']}")
                    meta = client.get_meta()
                    print(f"Version: {meta.get('version')}")
                    
                    # Verify gRPC by performing a dummy search (guaranteed to use gRPC in v4)
                    print("Verifying gRPC functionality with a query...")
                    try:
                        # This will throw a gRPC error if the channel is broken
                        dummy = client.collections.get("NonExistentCollection")
                        dummy.query.fetch_objects(limit=1)
                        print("gRPC query sent (received empty/not found as expected).")
                    except Exception as ge:
                        if "gRPC" in str(ge) or "Connect" in str(ge):
                            print(f"gRPC Specific Error: {ge}")
                        else:
                            # It's okay if it's just "collection not found" as long as it went through the channel
                            print(f"Query returned expected or manageable error: {ge}")
                    
                    print("gRPC verification completed.")
                    return
                else:
                    print(f"FAILED: Connection established but is_ready() is False")
            except Exception as e:
                print(f"FAILED: {type(e).__name__}: {e}")
            finally:
                client.close()

    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_weaviate_v4()
