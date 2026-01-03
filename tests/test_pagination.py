"""
Property-based tests for paginate_posts function.
Feature: forum-search-filter
"""
import math
import os
import sys

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import paginate_posts

# --- Generators ---

@st.composite
def simple_post_strategy(draw):
    """Generate a simple post for pagination testing."""
    return {
        'id': draw(st.integers(min_value=1, max_value=10000)),
        'content': draw(st.text(min_size=1, max_size=50)),
        'timestamp': '2025-01-01T00:00:00Z'
    }


@st.composite
def posts_list_strategy(draw):
    """Generate a list of posts."""
    return draw(st.lists(simple_post_strategy(), min_size=0, max_size=100))


# --- Property Tests ---

@settings(max_examples=100)
@given(
    posts=posts_list_strategy(),
    page=st.integers(min_value=1, max_value=20),
    per_page=st.integers(min_value=1, max_value=50)
)
def test_property_6_pagination_correctness(posts, page, per_page):
    """
    Property 6: Pagination Correctness
    For any list of posts, page number, and per_page value:
    - Results contain at most per_page posts
    - Correct slice of posts for given page
    - Accurate metadata (total, total_pages = ceil(total/per_page))
    
    Feature: forum-search-filter, Property 6: Pagination Correctness
    Validates: Requirements 4.4, 4.5, 4.6
    """
    result = paginate_posts(posts, page=page, per_page=per_page)
    
    total = len(posts)
    expected_total_pages = math.ceil(total / per_page) if total > 0 else 1
    
    # Check metadata accuracy
    assert result['total'] == total, f"Total should be {total}, got {result['total']}"
    assert result['per_page'] == per_page, f"per_page should be {per_page}"
    assert result['total_pages'] == expected_total_pages, \
        f"total_pages should be {expected_total_pages}, got {result['total_pages']}"
    
    # Check posts count <= per_page
    assert len(result['posts']) <= per_page, \
        f"Posts count {len(result['posts'])} exceeds per_page {per_page}"
    
    # Check page is within valid range
    assert 1 <= result['page'] <= expected_total_pages, \
        f"Page {result['page']} out of range [1, {expected_total_pages}]"


@settings(max_examples=100)
@given(
    posts=posts_list_strategy(),
    per_page=st.integers(min_value=1, max_value=50)
)
def test_pagination_covers_all_posts(posts, per_page):
    """
    Verify that paginating through all pages returns all posts exactly once.
    
    Feature: forum-search-filter, Property 6: Pagination Correctness
    Validates: Requirements 4.4, 4.5
    """
    if not posts:
        result = paginate_posts(posts, page=1, per_page=per_page)
        assert result['posts'] == []
        return
    
    all_paginated_posts = []
    total_pages = math.ceil(len(posts) / per_page)
    
    for page in range(1, total_pages + 1):
        result = paginate_posts(posts, page=page, per_page=per_page)
        all_paginated_posts.extend(result['posts'])
    
    # All posts should be covered exactly once
    assert len(all_paginated_posts) == len(posts), \
        f"Pagination should cover all {len(posts)} posts, got {len(all_paginated_posts)}"


@settings(max_examples=100)
@given(posts=posts_list_strategy(), per_page=st.integers(min_value=1, max_value=50))
def test_pagination_correct_slice(posts, per_page):
    """
    Verify that each page returns the correct slice of posts.
    
    Feature: forum-search-filter, Property 6: Pagination Correctness
    Validates: Requirements 4.4
    """
    if not posts:
        return
    
    total_pages = math.ceil(len(posts) / per_page)
    
    for page in range(1, total_pages + 1):
        result = paginate_posts(posts, page=page, per_page=per_page)
        
        # Calculate expected slice
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, len(posts))
        expected_posts = posts[start_idx:end_idx]
        
        assert result['posts'] == expected_posts, \
            f"Page {page} should return posts[{start_idx}:{end_idx}]"


@settings(max_examples=50)
@given(posts=posts_list_strategy())
def test_pagination_page_beyond_total(posts):
    """
    Verify behavior when requesting page beyond total pages.
    
    Feature: forum-search-filter, Property 6: Pagination Correctness
    Validates: Requirements 4.4
    """
    per_page = 10
    total_pages = math.ceil(len(posts) / per_page) if posts else 1
    
    # Request page way beyond total
    result = paginate_posts(posts, page=total_pages + 100, per_page=per_page)
    
    # Should clamp to last page
    assert result['page'] == total_pages, \
        f"Page should be clamped to {total_pages}, got {result['page']}"
