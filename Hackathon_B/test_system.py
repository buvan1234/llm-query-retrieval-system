#!/usr/bin/env python3
"""
Test script for the LLM-Powered Intelligent Query–Retrieval System.
This script demonstrates the system's capabilities with sample queries.
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
TEAM_TOKEN = "2bfa46dcffe69615807a8586036130853540f4c0682bcbbf5625c64a5f43e96f"
HEADERS = {
    "Authorization": f"Bearer {TEAM_TOKEN}",
    "Content-Type": "application/json"
}

# Sample test data
SAMPLE_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

SAMPLE_QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "What is the No Claim Discount (NCD) offered in this policy?",
    "Is there a benefit for preventive health check-ups?",
    "How does the policy define a 'Hospital'?",
    "What is the extent of coverage for AYUSH treatments?",
    "Are there any sub-limits on room rent and ICU charges for Plan A?"
]


def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()['status']}")
            print(f"   Message: {response.json()['message']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")


def test_query_processing():
    """Test the main query processing endpoint."""
    print("\n🔍 Testing query processing...")
    
    request_data = {
        "documents": SAMPLE_DOCUMENT_URL,
        "questions": SAMPLE_QUESTIONS[:3]  # Test with first 3 questions
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=HEADERS,
            json=request_data
        )
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Query processing successful")
            print(f"   Processing time: {processing_time:.2f} seconds")
            print(f"   Questions processed: {len(result['answers'])}")
            
            # Display answers
            for i, (question, answer) in enumerate(zip(request_data['questions'], result['answers']), 1):
                print(f"\n   Q{i}: {question}")
                print(f"   A{i}: {answer[:200]}...")
                
            # Display metadata if available
            if 'metadata' in result:
                metadata = result['metadata']
                print(f"\n   Metadata:")
                print(f"   - Total questions: {metadata.get('total_questions', 'N/A')}")
                print(f"   - Processing time: {metadata.get('processing_time', 'N/A')}")
                print(f"   - Document URL: {metadata.get('document_url', 'N/A')}")
                
        else:
            print(f"❌ Query processing failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Query processing error: {e}")


def test_stats_endpoint():
    """Test the stats endpoint."""
    print("\n🔍 Testing stats endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/stats", headers=HEADERS)
        if response.status_code == 200:
            stats = response.json()
            print("✅ Stats retrieved successfully")
            print(f"   Status: {stats['status']}")
            if 'stats' in stats:
                stats_data = stats['stats']
                print(f"   Current document: {stats_data.get('current_document', 'None')}")
                print(f"   Cache size: {stats_data.get('cache_size', 0)}")
        else:
            print(f"❌ Stats retrieval failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stats error: {e}")


def test_clauses_endpoint():
    """Test the clauses endpoint."""
    print("\n🔍 Testing clauses endpoint...")
    test_query = "grace period premium payment"
    
    try:
        response = requests.get(
            f"{BASE_URL}/clauses/{test_query}",
            headers=HEADERS,
            params={"top_k": 3}
        )
        if response.status_code == 200:
            result = response.json()
            print("✅ Clauses retrieval successful")
            print(f"   Query: {result['query']}")
            print(f"   Clauses found: {result['total_found']}")
            
            for i, clause in enumerate(result.get('clauses', [])[:2], 1):
                print(f"   Clause {i}: {clause.get('text', '')[:100]}...")
        else:
            print(f"❌ Clauses retrieval failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Clauses error: {e}")


def test_explanation_endpoint():
    """Test the explanation endpoint."""
    print("\n🔍 Testing explanation endpoint...")
    
    test_data = {
        "query": "What is the grace period for premium payment?",
        "answer": "A grace period of thirty days is provided for premium payment after the due date."
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/explain",
            headers=HEADERS,
            json=test_data
        )
        if response.status_code == 200:
            result = response.json()
            print("✅ Explanation generated successfully")
            print(f"   Status: {result['status']}")
            if 'explanation' in result:
                explanation = result['explanation']
                print(f"   Query: {explanation.get('query', 'N/A')}")
                print(f"   Answer: {explanation.get('answer', 'N/A')[:100]}...")
        else:
            print(f"❌ Explanation generation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Explanation error: {e}")


def run_performance_test():
    """Run a performance test with multiple questions."""
    print("\n🚀 Running performance test...")
    
    request_data = {
        "documents": SAMPLE_DOCUMENT_URL,
        "questions": SAMPLE_QUESTIONS  # All questions
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=HEADERS,
            json=request_data
        )
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Performance test completed")
            print(f"   Total time: {total_time:.2f} seconds")
            print(f"   Questions processed: {len(result['answers'])}")
            print(f"   Average time per question: {total_time/len(result['answers']):.2f} seconds")
            
            # Check answer quality
            valid_answers = sum(1 for answer in result['answers'] if answer and len(answer) > 10)
            print(f"   Valid answers: {valid_answers}/{len(result['answers'])}")
            
        else:
            print(f"❌ Performance test failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Performance test error: {e}")


def main():
    """Run all tests."""
    print("🧪 LLM-Powered Intelligent Query–Retrieval System Test Suite")
    print("=" * 60)
    
    # Test basic functionality
    test_health_check()
    test_query_processing()
    test_stats_endpoint()
    test_clauses_endpoint()
    test_explanation_endpoint()
    
    # Run performance test
    run_performance_test()
    
    print("\n" + "=" * 60)
    print("✅ Test suite completed!")


if __name__ == "__main__":
    main()
