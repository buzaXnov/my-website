import os
from datetime import datetime
from core.models import ParkingEvent, ParkingEventType, ParkingSpot, ParkingSpotType, Camera

# os.environ['DJANGO_SETTINGS_MODULE'] = 'myproject.myapp.settings'
# print(os.environ['DJANGO_SETTINGS_MODULE'])

# cam = Camera(name=, description=)
cam = Camera()
cam.save()

par_spot_type = ParkingSpotType(name=1)
par_spot_type.save()

par_spot = ParkingSpot(parking_spot_type=par_spot_type, camera=cam, yaml_file_location='/home/tmp/')
par_spot.save()

par_event_type = ParkingEventType(name=1)
par_event_type.save()

par_event = ParkingEvent(parking_spot=par_spot, parking_event_type=par_event_type, license_plate="ST321CA", time=datetime.now())
par_event.save()
