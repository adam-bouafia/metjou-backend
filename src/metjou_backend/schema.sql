-- Idempotent: applied at every start by db.init_schema().
create extension if not exists vector;

create table if not exists documents (
    id           bigserial primary key,
    url          text not null unique,
    title        text not null,
    organisation text not null,
    content_hash text not null,
    fetched_at   timestamptz not null default now()
);

create table if not exists passages (
    id          bigserial primary key,
    document_id bigint not null references documents (id) on delete cascade,
    position    int not null,
    text        text not null,
    embedding   vector(384) not null,
    tsv         tsvector generated always as (to_tsvector('dutch', text)) stored
);

create index if not exists passages_embedding_idx
    on passages using hnsw (embedding vector_cosine_ops);
create index if not exists passages_tsv_idx on passages using gin (tsv);
create index if not exists passages_document_idx on passages (document_id);
