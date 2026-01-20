from pathlib import Path
from fastapi import APIRouter

def get_asset_paths():
    base = Path(__file__).resolve().parent.parent / 'assets'
    return {
        'human_pic': base / 'human_pic',
        'DFM': base / 'DFM',
        'm_blank': base / 'm_blank',
    }

def get_resources_router() -> APIRouter:
    router = APIRouter()
    assets = get_asset_paths()

    def list_files(dir_key: str, mount_prefix: str):
        p = assets[dir_key]
        if not p.exists():
            return []
        result = []
        for f in sorted(p.iterdir()):
            if f.is_file():
                result.append(f"files/{mount_prefix}/{f.name}")
        return result

    @router.get('/human_pic')
    async def list_human_pic():
        return list_files('human_pic', 'human_pic')

    @router.get('/DFM')
    async def list_dfm():
        return list_files('DFM', 'DFM')

    @router.get('/m_blank')
    async def list_scene():
        return list_files('m_blank', 'm_blank')

    return router

