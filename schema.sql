-- Простая схема витрины продаж
CREATE TABLE IF NOT EXISTS sales (
    date        DATE        NOT NULL,
    store       INTEGER     NOT NULL,
    item        INTEGER     NOT NULL,
    sales       NUMERIC     NOT NULL,
    PRIMARY KEY (date, store, item)
);

-- Индексы для типичных аналитических запросов
CREATE INDEX IF NOT EXISTS idx_sales_store_item ON sales(store, item);
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date);