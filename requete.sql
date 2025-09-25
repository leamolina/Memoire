SET hive.execution.engine=tez;
SET tez.queue.name=SelfService;
SET hive.tez.container.size=8192;
SET hive.tez.java.opts=-Xmx6553m;
SET hive.server2.tez.sessions.per.default.queue=2;
SET hive.fetch.task.conversion=none;
SET hive.execution.engine=tez;
SET tez.queue.name=SelfService;
SET hive.tez.container.size=8192;
SET hive.tez.java.opts=-Xmx6553m;
SET hive.server2.tez.sessions.per.default.queue=2;
SET hive.fetch.task.conversion=none;

WITH filtered_data AS (
    SELECT
        ci_name,
        ci_type,
        event_date_time,
        event_type,
        device_role,
        CASE WHEN event_value > 100 THEN 100 ELSE event_value END AS event_value,
        site_longitude,
        site_latitude,
        service_line,
        site_name,
        site_city,
        site_country,
        site_criticality,
        site_business_hours
    FROM
        prod_usg_muxc.str_watchopenpollermetric
    WHERE
        event_type = 'in_bandwidth_pct'
        AND device_type_label = 'IP Router'
        AND YEAR(event_date_time) = 2025
        AND MONTH(event_date_time) = 6
        --AND HOUR(event_date_time) >= 7 AND HOUR(event_date_time) < 18
        AND customer_name = 'LAPOSTE'
),
daily_hourly_max AS (
    SELECT
        ci_name,
        ci_type,
        device_role,
        site_longitude,
        site_latitude,
        service_line,
        site_name,
        site_city,
        site_country,
        site_criticality,
        site_business_hours,
        DAY(event_date_time) AS event_day,
        HOUR(event_date_time) AS event_hour,
        MAX(event_value) AS max_event_value
    FROM
        filtered_data
    GROUP BY
        ci_name,
        ci_type,
        device_role,
        site_longitude,
        site_latitude,
        service_line,
        site_name,
        site_city,
        site_country,
        site_criticality,
        site_business_hours,
        DAY(event_date_time),
        HOUR(event_date_time) 
)
SELECT
    ci_name,
    ci_type,
    device_role,
    site_longitude,
    site_latitude,
    service_line,
    site_name,
    site_city,
    site_country,
    site_criticality,
    site_business_hours,
    event_day,
    event_hour,
    max_event_value
FROM
    daily_hourly_max
ORDER BY
    ci_name, 
    event_day, 
    event_hour;