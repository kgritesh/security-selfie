# -*- coding: utf-8 -*-
import aiofiles
import hupper
from sanic import Sanic
from sanic import response

from blink_detector import BlinkDetector


def create_app():
    app = Sanic(__name__)
    app.static('/static', './static')
    register_routes(app)
    return app


def register_routes(app):
    @app.route('/')
    async def index(request):
        async with aiofiles.open('./static/index.html') as f:
            content = await f.read()

        return response.html(content)

    @app.route('/upload', methods=['POST'])
    async def handle_upload(request):
        video = request.files.get('video')
        form_data = request.form
        async with aiofiles.open('./data/video.mp4', 'wb') as f:
            await f.write(video.body)

        blinks = [int(s) for s in form_data.get('blinks').split(',')]
        b = BlinkDetector('shape_predictor_68_face_landmarks.dat', './data/video.mp4')
        b.start()
        print('Got Blink:{}, detected blinks: {}'.format(blinks, b.blinks))
        status = 'success' if blinks == b.blinks else 'failure'
        return response.json({
            'status': status
        })


def run_app():
    app = create_app()
    app.run(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    hupper.start_reloader('app.run_app')
