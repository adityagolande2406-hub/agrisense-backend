from app.routers.auth import router as auth_router
from app.routers.farms import farms_router, fields_router, crops_router
from app.routers.observations import router as observations_router
from app.routers.predict import router as predict_router
from app.routers.alerts import router as alerts_router
from app.routers.dashboard import router as dashboard_router

__all__ = [
    "auth_router",
    "farms_router",
    "fields_router",
    "crops_router",
    "observations_router",
    "predict_router",
    "alerts_router",
    "dashboard_router",
]
