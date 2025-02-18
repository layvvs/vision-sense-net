from config import get_config
from inference_app import InferenceApp


def main():
    config = get_config()
    app = InferenceApp(config)

    try:
        app.run()
    except KeyboardInterrupt:
        print('stop main')
        pass


if __name__ == '__main__':
    main()
