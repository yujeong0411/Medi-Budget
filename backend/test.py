import psycopg2

try:
    conn = psycopg2.connect(
        host="121.178.98.98",  # 서버 공인 IP 또는 도메인
        port="5432",       # PostgreSQL 기본 포트
        database="medibudget",
        user="postgres",
        password="dbwjd0411!"
    )
    print("연결 성공!")
    conn.close()
except Exception as e:
    print(f"연결 실패: {e}")