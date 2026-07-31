import redis

def explore_redis():
    try:
        # Connect to Redis without automatic decoding to handle binary vectors safely
        r = redis.Redis.from_url("redis://localhost:6379", decode_responses=False)
        
        if not r.ping():
            print("Cannot connect to Redis")
            return
        print("Successfully connected to Redis!\n")
        
        # 1. List keys
        keys_bytes = list(r.scan_iter(match=b"*", count=100))
        keys = [k.decode('utf-8', errors='ignore') for k in keys_bytes]
        
        print(f"Total keys in Redis (showing up to 100): {len(keys)}")
        if not keys:
            print("Redis is empty. Run your QA bot and upload a PDF to populate it.")
            return
            
        print("Here are up to 10 keys currently in your Redis instance:")
        for k in keys[:10]:
            print(f" - {k}")
            
        print("\n" + "="*50 + "\n")
        
        # 2. Look for vector chunks
        doc_keys = [k for k in keys_bytes if b"qa_chatbot_index" in k and r.type(k) == b'hash']
        
        if doc_keys:
            print(f"Found {len(doc_keys)} document chunks saved in the vector store!")
            print("Let's look at the first document chunk:\n")
            
            sample_doc_key = doc_keys[0]
            doc_data = r.hgetall(sample_doc_key)
            
            print(f"Key: {sample_doc_key.decode('utf-8', errors='ignore')}")
            
            if b'content' in doc_data:
                content_str = doc_data[b'content'].decode('utf-8', errors='ignore')
                print(f"Text Content snippet:\n{content_str[:200]}...\n")
                
            if b'content_vector' in doc_data:
                 print(f"Vector Embedding found: Yes (Binary data omitted for readability)")
        else:
            print("No vector store documents found. Have you uploaded a PDF via the Gradio app yet?")
            
    except Exception as e:
        print(f"Error connecting to Redis: {e}")

if __name__ == "__main__":
    explore_redis()
