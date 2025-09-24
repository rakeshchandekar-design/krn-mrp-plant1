from fastapi import APIRouter

router = APIRouter(prefix='/grn', tags=['GRN'])

@router.get('/')
def list_grns():
    return {'msg': 'List GRNs (placeholder)'}
