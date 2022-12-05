from datetime import datetime
from app.core.models import ParkingEvent

par_event = ParkingEvent(license_plate="ZG123AC", time=datetime.now())

par_event.save()
