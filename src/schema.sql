-- Table: flickr_data

-- DROP TABLE flickr_data;

CREATE TABLE flickr_data
(
  ispublic integer,
  place_id character varying,
  geo_is_public integer,
  owner character varying,
  id character varying,
  title text,
  woeid character varying,
  geo_is_friend integer,
  geo_is_contact integer,
  datetaken character varying,
  isfriend integer,
  secret character varying,
  ownername character varying,
  latitude double precision,
  longitude double precision,
  accuracy character varying,
  isfamily integer,
  tags text,
  farm integer,
  geo_is_family integer,
  dateupload character varying,
  datetakengranularity integer,
  server character varying,
  context integer,
  internal_id integer NOT NULL DEFAULT nextval('id_seq'::regclass),
  CONSTRAINT flickr_data_pkey PRIMARY KEY (internal_id)
)
WITH (
  OIDS=FALSE
);
ALTER TABLE flickr_data
  OWNER TO jimmy1;


CREATE sequence class_seq;

drop table classifications;

CREATE TABLE IF NOT EXISTS classifications (
 	pred_code INTEGER,
 	fl_internal_id INTEGER,
 	notes TEXT,
 	classrun INTEGER,
 	latitude double precision,
 	longitude double precision,
 	id integer  NOT NULL DEFAULT nextval('class_seq'::regclass),
 	CONSTRAINT classifications_pkey PRIMARY KEY (id)
)

CREATE SEQUENCE geo_class_id;

CREATE TABLE geo_class (
  geo_code INTEGER,
  geo_text VARCHAR,
  internal_id INTEGER,
  class_notes VARCHAR,
  id integer  NOT NULL DEFAULT nextval('geo_class_id'::regclass),
  CONSTRAINT geo_class_id_pkey PRIMARY KEY (id)
)