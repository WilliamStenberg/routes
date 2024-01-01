from dashboard import Model
import db as db
import importfiles as importfiles

if __name__ == '__main__':
    engine = db.make_engine()
    importfiles.sync(engine)
    model = Model(db_engine=engine)
    model.run_server()
