CREATE DATABASE smart_parking;

CREATE TABLE parking_event_type(
    id serial,
    name VARCHAR(25),
    PRIMARY KEY(id)
);

INSERT INTO parking_event_type(name) VALUES ('Parking space occupied.');
INSERT INTO parking_event_type(name) VALUES ('Parking space released.');

CREATE TABLE camera(
    id serial,
    name VARCHAR(50),
    description VARCHAR(255),
    PRIMARY KEY(id)
);

INSERT INTO camera(name, description) VALUES ('Test camera', 'Located onsite.');

CREATE TABLE parking_spot_type(
    id serial,
    name VARCHAR(50),
    PRIMARY KEY(id)
);

INSERT INTO parking_spot_type(name) VALUES('Payed parking spot.');
INSERT INTO parking_spot_type(name) VALUES('Disabled parking spot.');
INSERT INTO parking_spot_type(name) VALUES('Car entrance.');

CREATE TABLE parking_spot(
    id serial,
    parking_spot_type_id INT NOT NULL,
    camera_id INT NOT NULL,
    yaml_file_location VARCHAR(100),
    PRIMARY KEY(id),
    FOREIGN KEY (parking_spot_type_id)
		REFERENCES parking_spot_type(id),
    FOREIGN KEY (camera_id)
		REFERENCES camera(id)
);

CREATE TABLE parking_event(
    id serial,
    parking_spot_id INT NOT NULL,
    parking_event_type_id INT NOT NULL,
    license_plate VARCHAR(15) NOT NULL,
    time TIMESTAMP NOT NULL,
    PRIMARY KEY(id),
    FOREIGN KEY (parking_event_type_id)
        REFERENCES parking_event_type(id),
    FOREIGN KEY (parking_spot_id)
        REFERENCES parking_spot(id)
);
