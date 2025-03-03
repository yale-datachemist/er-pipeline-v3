import weaviate

# Initialize the Weaviate client
client = weaviate.connect_to_local()

def explore_collections():
    """
    Fetch and display information about collections in the Weaviate instance.
    """
    try:
        schema = client.collections.get("UniqueStrings")
        print(schema)


    except Exception as e:
        print(f"Error fetching schema: {e}")

    finally:
        client.close()  # Ensure the client is properly closed

if __name__ == "__main__":
    explore_collections()
