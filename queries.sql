-- Топ-10 товаров по средним продажам за последние 30 дней
WITH recent AS (
  SELECT * FROM sales WHERE date >= DATE('now', '-30 day')
)
SELECT item, AVG(sales) AS avg_sales
FROM recent
GROUP BY item
ORDER BY avg_sales DESC
LIMIT 10;

-- Динамика по магазину/товару с недельной агрегацией
SELECT store, item, strftime('%Y-%W', date) AS year_week, SUM(sales) AS weekly_sales
FROM sales
GROUP BY store, item, year_week
ORDER BY store, item, year_week;

-- Витрина с календарными признаками
SELECT
  date,
  store,
  item,
  sales,
  CAST (strftime('%w', date) AS INT) AS dow,
  CAST (strftime('%m', date) AS INT) AS month
FROM sales;