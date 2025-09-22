from fastapi import APIRouter


router = APIRouter()


@router.get("/results/{analysis_id}")
async def get_results(analysis_id: str):
    # Stub: In the future, this will load results from persistence.
    return {
        "analysis_id": analysis_id,
        "status": "pending",
        "message": "Stub: results retrieval will be implemented",
    }
