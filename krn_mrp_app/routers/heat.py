from fastapi import APIRouter

router = APIRouter(prefix='/heat', tags=['Heat'])

@router.get('/')
def list_heats():
    return {'msg': 'List Heats (placeholder)'}
