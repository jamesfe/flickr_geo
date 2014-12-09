
CREATE SEQUENCE id_seq;

 CREATE TABLE IF NOT EXISTS flickr_data (
 	ispublic INT,
 	place_id VARCHAR,
 	geo_is_public INT,
 	owner VARCHAR,
 	id VARCHAR, 
 	title TEXT,
 	woeid VARCHAR,
 	geo_is_friend INT,
 	geo_is_contact INT,
 	datetaken VARCHAR,
 	isfriend INT,
 	secret VARCHAR,
 	ownername VARCHAR,
 	latitude FLOAT,
 	longitude FLOAT,
 	accuracy VARCHAR,
 	isfamily INT,
 	tags TEXT,
 	farm INT,
 	geo_is_family INT,
 	dateupload VARCHAR,
 	datetakengranularity INT,
 	server VARCHAR,
 	context	INT
 	"id" integer PRIMARY KEY default nextval('id_seq')
 	);