"""Default service endpoints for the server environment."""

SAM3_SEGMENT_URL = "http://101.132.143.105:5081/segment"
SAM3D_PROCESS_URL = "http://101.132.143.105:5077/api/process"
MOGE_URL = "http://101.132.143.105:5014"
S2M2_URLS = [f"http://10.10.4.244:{port}/api/process" for port in range(5060, 5068)]
KNOWIN_WORLD_PYTHON = "/mnt/workspace/wenqian/knowin-world/.venv/bin/python"
