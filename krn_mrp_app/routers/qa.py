from fastapi import APIRouter

router = APIRouter(prefix='/qa', tags=['QA'])

@router.get('/')
def list_qas():
    return {'msg': 'List QA approvals (placeholder)'}
