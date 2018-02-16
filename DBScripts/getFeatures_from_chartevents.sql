
/*
Following View contails min, max, mean, std_dev and numbers of measurements of each itemid for each icustay_id
*/
DROP MATERIALIZED VIEW IF EXISTS CHARTSMEASURMENTS CASCADE;
CREATE MATERIALIZED VIEW CHARTSMEASURMENTS AS

select icustay_id,itemid, avg(valuenum) as mean_val, max(valuenum) as max_val, min(valuenum) as min_val, stddev(valuenum) as stddev_val, count(valuenum) as count_val
from Chartevents
where (icustay_id is not null) and (itemid is not null) and (valuenum is not null)
group by icustay_id,itemid
order by icustay_id;

/*
CHARTSLASTMSMTS view contails all the last measured events for Heart Rate, Respiratory Rate, Systolic and Diastolic BPs (Mean, ABP, NBP)
, ABP). This view was seperated from other features of charevents because of huge size of chartevents data
*/

DROP MATERIALIZED VIEW IF EXISTS CHARTSLASTMSMTS CASCADE;
CREATE MATERIALIZED VIEW CHARTSLASTMSMTS AS
WITH LatestChartimes AS (
select icustay_id,itemid, max(charttime) as max_charttime
from Chartevents
where (icustay_id is not null) and (valuenum is not null)
and itemid in (211, 220045, 618, 220210, 676, 678, 223761, 223762,51, 220050, 6, 6701, 225309, 455, 220179, 8364, 8368, 8555, 220051, 225310, 8441, 220180,224, 52, 6702, 6927, 220052, 456, 220181)
group by icustay_id,itemid
)
select chrt.icustay_id, chrt.itemid, chrt.valuenum as lastmsmt
from Chartevents chrt, LatestChartimes lchrt
where 
chrt.itemid = lchrt.itemid and
chrt.icustay_id = lchrt.icustay_id and
chrt.charttime = lchrt.max_charttime;