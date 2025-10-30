"""
Vercel Serverless Function Entry Point
This file adapts the Flask app for Vercel's serverless environment
"""

from src.app import app

# Vercel serverless function handler
def handler(event, context):
    """
    Handler function for Vercel serverless deployment
    """
    return app
