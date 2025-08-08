#!/usr/bin/env python3
"""
Demo script for the LLM-Powered Intelligent Query–Retrieval System.
This script demonstrates the system's capabilities with a simple example.
"""

import requests
import json
import time

def demo_system():
    """Demonstrate the system's capabilities."""
    print("🎯 LLM-Powered Intelligent Query–Retrieval System Demo")
    print("=" * 60)
    
    # Configuration
    base_url = "http://localhost:8000/api/v1"
    headers = {
        "Authorization": "Bearer 2bfa46dcffe69615807a8586036130853540f4c0682bcbbf5625c64a5f43e96f",
        "Content-Type": "application/json"
    }
    
    # Sample data
    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
    ]
    
    print("📋 Sample Questions:")
    for i, question in enumerate(questions, 1):
        print(f"   {i}. {question}")
    
    print("\n🔄 Processing...")
    print("   - Downloading document")
    print("   - Extracting text and creating chunks")
    print("   - Generating embeddings")
    print("   - Building search index")
    print("   - Processing queries with LLM")
    
    # Make the request
    request_data = {
        "documents": document_url,
        "questions": questions
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/hackrx/run",
            headers=headers,
            json=request_data
        )
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ Processing completed in {processing_time:.2f} seconds")
            print("\n📝 Answers:")
            
            for i, (question, answer) in enumerate(zip(questions, result['answers']), 1):
                print(f"\n   Q{i}: {question}")
                print(f"   A{i}: {answer}")
            
            # Show metadata
            if 'metadata' in result:
                metadata = result['metadata']
                print(f"\n📊 Processing Statistics:")
                print(f"   - Total questions: {metadata.get('total_questions', 'N/A')}")
                print(f"   - Processing time: {metadata.get('processing_time', 'N/A')} seconds")
                print(f"   - Document URL: {metadata.get('document_url', 'N/A')[:50]}...")
                
                if 'index_stats' in metadata:
                    stats = metadata['index_stats']
                    print(f"   - Total chunks: {stats.get('total_chunks', 'N/A')}")
                    print(f"   - Index size: {stats.get('index_size', 'N/A')}")
                    print(f"   - Embedding dimension: {stats.get('dimension', 'N/A')}")
            
            print("\n🎉 Demo completed successfully!")
            print("\n💡 System Features Demonstrated:")
            print("   ✅ Document processing (PDF)")
            print("   ✅ Text extraction and chunking")
            print("   ✅ Semantic search with embeddings")
            print("   ✅ LLM-powered answer generation")
            print("   ✅ Structured JSON responses")
            print("   ✅ Processing metadata and statistics")
            
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Make sure the server is running!")
        print("   Start the server with: python start_server.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def show_system_info():
    """Show system information and available endpoints."""
    print("\n🔧 System Information:")
    print("   - API Base URL: http://localhost:8000/api/v1")
    print("   - Documentation: http://localhost:8000/docs")
    print("   - Health Check: http://localhost:8000/api/v1/health")
    print("\n📚 Available Endpoints:")
    print("   - POST /hackrx/run - Process documents and answer questions")
    print("   - GET /stats - Get processing statistics")
    print("   - GET /clauses/{query} - Get relevant policy clauses")
    print("   - POST /explain - Generate explanation for decision")
    print("   - POST /clear-cache - Clear processing cache")


if __name__ == "__main__":
    demo_system()
    show_system_info()
