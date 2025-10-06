#!/usr/bin/env python3
"""
Test script to verify the skill gap analysis functionality
"""

import requests
import json

def test_gap_analysis():
    base_url = "http://127.0.0.1:5000"
    
    # Test status endpoint
    print("Testing status endpoint...")
    try:
        response = requests.get(f"{base_url}/status")
        if response.status_code == 200:
            status_data = response.json()
            print("✅ Status endpoint working")
            print(f"   Model: {status_data.get('model_name')}")
            print(f"   Skill Gap Analysis: {status_data.get('skill_gap_analysis')}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing status: {e}")
    
    # Test recommendations for an existing user
    print("\nTesting recommendations with skill gap analysis...")
    try:
        # Try to get recommendations for user 14 (from logs)
        response = requests.get(f"{base_url}/recommendations/14")
        if response.status_code == 200:
            rec_data = response.json()
            print("✅ Recommendations endpoint working")
            
            if 'recommendations' in rec_data and rec_data['recommendations']:
                first_rec = rec_data['recommendations'][0]
                print(f"   First recommendation: {first_rec.get('name', 'Unknown')}")
                print(f"   Match score: {first_rec.get('match_score', 'N/A')}%")
                
                # Check for gap analysis fields
                gap_fields = ['matching_skills', 'missing_skills', 'skills_to_improve', 'learning_recommendations']
                has_gap_analysis = any(field in first_rec for field in gap_fields)
                
                if has_gap_analysis:
                    print("✅ Skill gap analysis fields present")
                    if 'matching_skills' in first_rec:
                        print(f"   Matching skills: {first_rec['matching_skills']}")
                    if 'missing_skills' in first_rec:
                        print(f"   Missing skills: {first_rec['missing_skills']}")
                    if 'learning_recommendations' in first_rec:
                        print(f"   Learning recommendations: {len(first_rec['learning_recommendations'])} items")
                else:
                    print("❌ Skill gap analysis fields missing")
            else:
                print("❌ No recommendations found")
        else:
            print(f"❌ Recommendations endpoint failed: {response.status_code}")
            if response.status_code == 404:
                print("   User 14 not found, try uploading a CV first")
    except Exception as e:
        print(f"❌ Error testing recommendations: {e}")

if __name__ == "__main__":
    test_gap_analysis()