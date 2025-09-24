from fastapi import APIRouter

router = APIRouter(prefix='/lot', tags=['Lot'])

@router.get('/')
def list_lots():
    return {'msg': 'List Lots (placeholder)'}
