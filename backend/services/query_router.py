"""Backward-compatible import wrapper.

The canonical router lives in backend.chains.query_router.
"""

from backend.chains.query_router import QueryRoute, route_query

__all__ = ["QueryRoute", "route_query"]
