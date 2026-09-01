from pathlib import Path

import psycopg2



DATA_DIR = Path("data")


CATEGORIES = [

    "bottle",

    "metal_nut",

    "screw",

]



def get_connection():

    return psycopg2.connect(

        host="localhost",

        port=5432,

        dbname="anomaly_db",

        user="anomaly_user",

        password="anomaly_password",

    )



def collect_images():

    records = []


    for category in CATEGORIES:

        category_path = DATA_DIR / category


        for split in ["train", "test"]:

            split_path = category_path / split


            if not split_path.exists():

                continue


            for defect_dir in split_path.iterdir():

                if not defect_dir.is_dir():

                    continue


                defect_type = defect_dir.name

                is_anomaly = defect_type != "good"


                for image_path in defect_dir.glob("*.png"):

                    records.append(

                        (

                            str(image_path),

                            category,

                            split,

                            defect_type,

                            is_anomaly,

                        )

                    )


    return records



def insert_records(records):

    connection = get_connection()

    cursor = connection.cursor()


    query = """

        INSERT INTO images (

            image_path,

            category,

            split,

            defect_type,

            is_anomaly

        )

        VALUES (%s, %s, %s, %s, %s)

        ON CONFLICT (image_path) DO NOTHING;

    """


    cursor.executemany(query, records)


    connection.commit()


    cursor.close()

    connection.close()



if __name__ == "__main__":

    records = collect_images()


    print(f"Found {len(records)} images.")


    insert_records(records)


    print("Metadata successfully inserted into PostgreSQL.")
