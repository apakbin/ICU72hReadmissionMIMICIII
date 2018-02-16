
/*
Following View contails min, max, mean, std_dev and numbers of measurements of each itemid for each icustay_id from labevents
*/
DROP MATERIALIZED VIEW IF EXISTS LABMEASURMENTS CASCADE;
CREATE MATERIALIZED VIEW LABMEASURMENTS AS

select le.hadm_id, icustay_id, le.itemid, avg(valuenum) as mean_val, max(valuenum) as max_val ,
 min(valuenum) as min_val, stddev(valuenum) as stddev_val, count(valuenum) as count_val
from LABEVENTS le
inner join ICUSTAYS ics
on ics.hadm_id=le.hadm_id
and le.charttime BETWEEN ics.intime and ics.outtime
where (icustay_id is not null) and (le.itemid is not null) and (le.valuenum is not null)
group by le.hadm_id,icustay_id,le.itemid
order by icustay_id;


/*
LABSLASTMSMTS view contails all the last measured events for obeservations on labevents
*/
DROP MATERIALIZED VIEW IF EXISTS LABSLASTMSMTS CASCADE;
CREATE MATERIALIZED VIEW LABSLASTMSMTS AS

select t1.max_charttime, t1.icustay_id, t1.itemid, t2.valuenum
from

(select le.hadm_id, icustay_id, le.itemid, max(charttime) as max_charttime
from LABEVENTS le
inner join ICUSTAYS ics
on ics.hadm_id=le.hadm_id
and le.charttime BETWEEN ics.intime and ics.outtime
where (icustay_id is not null) and (le.itemid is not null) and (le.valuenum is not null)
group by le.hadm_id,icustay_id,le.itemid
order by icustay_id) t1

inner join LABEVENTS t2

on t1.hadm_id = t2.hadm_id
and t1.max_charttime = t2.charttime
and t1.itemid = t2.itemid;