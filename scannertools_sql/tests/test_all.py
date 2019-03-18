import pytest
from testing.postgresql import Postgresql
import psycopg2
import scannerpy
import json
from scannertools_infra.tests import sc
from scannerpy import protobufs
import scannertools_sql
from scannertools_sql.storage import SQLStorage, SQLInputStream, SQLOutputStream

@pytest.fixture(scope='module')
def sql_sc(sc):
    with Postgresql() as postgresql:
        conn = psycopg2.connect(**postgresql.dsn())
        cur = conn.cursor()

        cur.execute(
            'CREATE TABLE test (id serial PRIMARY KEY, a integer, b integer, c text, d varchar(255), e boolean, f float, grp integer)'
        )
        cur.execute(
            "INSERT INTO test (a, b, c, d, e, f, grp) VALUES (10, 0, 'hi', 'hello', true, 2.0, 0)"
        )
        cur.execute(
            "INSERT INTO test (a, b, c, d, e, f, grp) VALUES (20, 0, 'hi', 'hello', true, 2.0, 0)"
        )
        cur.execute(
            "INSERT INTO test (a, b, c, d, e, f, grp) VALUES (30, 0, 'hi', 'hello', true, 2.0, 1)"
        )
        cur.execute('CREATE TABLE jobs (id serial PRIMARY KEY, name text)')
        cur.execute(
            'CREATE TABLE test2 (id serial PRIMARY KEY, b integer, s text)')
        conn.commit()

        sql_params = postgresql.dsn()
        sql_config = protobufs.SQLConfig(
            hostaddr=sql_params['host'],
            port=sql_params['port'],
            dbname=sql_params['database'],
            user=sql_params['user'],
            adapter='postgres')

        yield sc, SQLStorage(config=sql_config, job_table='jobs'), cur

        cur.close()
        conn.close()


@scannerpy.register_python_op(name='AddOne')
def add_one(config, row: bytes) -> bytes:
    row = json.loads(row.decode('utf-8'))
    return json.dumps([{'id': r['id'], 'b': r['a'] + 1} for r in row])


def test_sql(sql_sc):
    (sc, storage, cur) = sql_sc

    cur.execute('SELECT COUNT(*) FROM test');
    n, = cur.fetchone()

    row = sc.io.Input([SQLInputStream(
        query=protobufs.SQLQuery(
            fields='test.id as id, test.a, test.c, test.d, test.e, test.f',
            table='test',
            id='test.id',
            group='test.id'),
        filter='true',
        storage=storage,
        num_elements=n)])
    row2 = sc.ops.AddOne(row=row)
    output_op = sc.io.Output(row2, [SQLOutputStream(
        table='test',
        storage=storage,
        job_name='foobar',
        insert=False)])
    sc.run(output_op)

    cur.execute('SELECT b FROM test')
    assert cur.fetchone()[0] == 11

    cur.execute('SELECT name FROM jobs')
    assert cur.fetchone()[0] == 'foobar'

@scannerpy.register_python_op(name='AddAll')
def add_all(config, row: bytes) -> bytes:
    row = json.loads(row.decode('utf-8'))
    total = sum([r['a'] for r in row])
    return json.dumps([{'id': r['id'], 'b': total} for r in row])


def test_sql_grouped(sql_sc):
    (sc, storage, cur) = sql_sc

    row = sc.io.Input([SQLInputStream(
        storage=storage,
        query=protobufs.SQLQuery(
            fields='test.id as id, test.a',
            table='test',
            id='test.id',
            group='test.grp'),
        filter='true')])
    row2 = sc.ops.AddAll(row=row)
    output_op = sc.io.Output(
        row2, [SQLOutputStream(storage=storage, table='test', job_name='test', insert=False)])
    sc.run(output_op)

    cur.execute('SELECT b FROM test')
    assert cur.fetchone()[0] == 30


@scannerpy.register_python_op(name='SQLInsertTest')
def sql_insert_test(config, row: bytes) -> bytes:
    row = json.loads(row.decode('utf-8'))
    return json.dumps([{'s': 'hello world', 'b': r['a'] + 1} for r in row])


def test_sql_insert(sql_sc):
    (sc, storage, cur) = sql_sc

    row = sc.io.Input([SQLInputStream(
        storage=storage,
        query=protobufs.SQLQuery(
            fields='test.id as id, test.a',
            table='test',
            id='test.id',
            group='test.grp'),
        filter='true')])
    row2 = sc.ops.SQLInsertTest(row=row)
    output_op = sc.io.Output(
        row2, [SQLOutputStream(
            storage=storage, table='test2', job_name='test', insert=True)])
    sc.run(output_op, show_progress=False)

    cur.execute('SELECT s FROM test2')
    assert cur.fetchone()[0] == "hello world"
