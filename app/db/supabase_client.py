from supabase import create_client, Client
import os

SUPABASE_URL = "https://kpscbcgyymjtqhpbwofx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imtwc2NiY2d5eW1qdHFocGJ3b2Z4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1OTQ4ODk5MiwiZXhwIjoyMDc1MDY0OTkyfQ.EtdMzyi3w21KFvWvXeJdaTs3lqELCQwAHPxpruQXmww"



supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

