"""
Property-based tests for filter_posts function.
Feature: forum-search-filter
"""
import os
import sys
from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import filter_posts, parse_post_timestamp

# --- Generators ---

def generate_timestamp(dt):
    """Convert datetime to ISO timestamp string."""
    return dt.isoformat() + 'Z'


@st.composite
def post_strategy(draw):
    """Generate a random post with content and timestamp."""
    content = draw(st.text(min_size=1, max_size=200))
    # Generate timestamp within last 2 years
    days_ago = draw(st.integers(min_value=0, max_value=730))
    timestamp = datetime.utcnow() - timedelta(days=days_ago)
    return {
        'id': draw(st.integers(min_value=1, max_value=10000)),
        'content': content,
        'timestamp': generate_timestamp(timestamp)
    }


@st.composite
def posts_list_strategy(draw):
    """Generate a list of random posts."""
    return draw(st.lists(post_strategy(), min_size=0, max_size=50))


@st.composite
def date_range_strategy(draw):
    """Generate a valid date range (start <= end)."""
    days_ago_start = draw(st.integers(min_value=0, max_value=730))
    days_ago_end = draw(st.integers(min_value=0, max_value=days_ago_start))
    start_date = datetime.utcnow() - timedelta(days=days_ago_start)
    end_date = datetime.utcnow() - timedelta(days=days_ago_end)
    return start_date, end_date


# --- Property Tests ---

@settings(max_examples=100)
@given(posts=posts_list_strategy(), search=st.text(min_size=1, max_size=20))
def test_property_1_search_filter_correctness(posts, search):
    """
    Property 1: Search Filter Correctness
    For any list of posts and any non-empty search query, all posts returned 
    by the filter function SHALL contain the search term (case-insensitive).
    
    Feature: forum-search-filter, Property 1: Search Filter Correctness
    Validates: Requirements 1.1, 1.4, 4.1
    """
    result = filter_posts(posts, search=search)
    search_lower = search.lower()
    
    for post in result:
        assert search_lower in post['content'].lower(), \
            f"Post content '{post['content']}' does not contain search term '{search}'"


@settings(max_examples=100)
@given(posts=posts_list_strategy())
def test_property_2_empty_filter_returns_all(posts):
    """
    Property 2: Empty Filter Returns All Posts
    For any list of posts, when no search query and no date filters are applied,
    the filter function SHALL return all posts unchanged.
    
    Feature: forum-search-filter, Property 2: Empty Filter Returns All Posts
    Validates: Requirements 1.2, 2.4
    """
    result = filter_posts(posts, search=None, start_date=None, end_date=None)
    assert len(result) == len(posts), "Empty filter should return all posts"
    
    # Also test with empty string search
    result_empty_search = filter_posts(posts, search='', start_date=None, end_date=None)
    assert len(result_empty_search) == len(posts), "Empty string search should return all posts"


@settings(max_examples=100)
@given(posts=posts_list_strategy(), date_range=date_range_strategy())
def test_property_3_start_date_filter_correctness(posts, date_range):
    """
    Property 3: Start Date Filter Correctness
    For any list of posts and any valid start date, all posts returned 
    SHALL have timestamps >= start_date.
    
    Feature: forum-search-filter, Property 3: Start Date Filter Correctness
    Validates: Requirements 2.1, 4.2
    """
    start_date, _ = date_range
    result = filter_posts(posts, start_date=start_date)
    
    for post in result:
        post_time = parse_post_timestamp(post['timestamp'])
        assert post_time >= start_date, \
            f"Post timestamp {post_time} is before start_date {start_date}"


@settings(max_examples=100)
@given(posts=posts_list_strategy(), date_range=date_range_strategy())
def test_property_4_end_date_filter_correctness(posts, date_range):
    """
    Property 4: End Date Filter Correctness
    For any list of posts and any valid end date, all posts returned 
    SHALL have timestamps <= end_date.
    
    Feature: forum-search-filter, Property 4: End Date Filter Correctness
    Validates: Requirements 2.2, 4.3
    """
    _, end_date = date_range
    result = filter_posts(posts, end_date=end_date)
    
    for post in result:
        post_time = parse_post_timestamp(post['timestamp'])
        assert post_time <= end_date, \
            f"Post timestamp {post_time} is after end_date {end_date}"


@settings(max_examples=100)
@given(posts=posts_list_strategy(), search=st.text(min_size=1, max_size=10), date_range=date_range_strategy())
def test_property_5_combined_filter_intersection(posts, search, date_range):
    """
    Property 5: Combined Filter Intersection
    For any list of posts, search query, and date range, the filtered results 
    SHALL be the intersection of posts matching search AND within date range.
    
    Feature: forum-search-filter, Property 5: Combined Filter Intersection
    Validates: Requirements 3.1, 3.2
    """
    start_date, end_date = date_range
    
    # Get combined filter result
    combined_result = filter_posts(posts, search=search, start_date=start_date, end_date=end_date)
    
    # Get individual filter results
    search_result = set(p['id'] for p in filter_posts(posts, search=search))
    date_result = set(p['id'] for p in filter_posts(posts, start_date=start_date, end_date=end_date))
    
    # Combined should be intersection
    expected_ids = search_result & date_result
    actual_ids = set(p['id'] for p in combined_result)
    
    assert actual_ids == expected_ids, \
        f"Combined filter should be intersection. Expected {expected_ids}, got {actual_ids}"


@settings(max_examples=100)
@given(posts=posts_list_strategy(), search=st.text(min_size=1, max_size=20))
def test_property_7_case_insensitivity(posts, search):
    """
    Property 7: Case Insensitivity
    For any post content and search term, searching with the term in any case 
    combination SHALL return the same results.
    
    Feature: forum-search-filter, Property 7: Case Insensitivity
    Validates: Requirements 1.4
    """
    result_lower = filter_posts(posts, search=search.lower())
    result_upper = filter_posts(posts, search=search.upper())
    result_original = filter_posts(posts, search=search)
    
    # All should return same posts (by id)
    ids_lower = set(p['id'] for p in result_lower)
    ids_upper = set(p['id'] for p in result_upper)
    ids_original = set(p['id'] for p in result_original)
    
    assert ids_lower == ids_upper == ids_original, \
        "Search should be case-insensitive"
