import tornado.ioloop
import tornado.web


is_running = False


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


class LaunchHandler(tornado.web.RequestHandler):
    """docstring for LaunchHandler"""
    def get(self):
        self.render("templates/launchform.html", title="Send Job")
        pass

    def post(self):
        pass


class StatusHandler(tornado.web.RequestHandler):
    """docstring for StatusHandler"""

    def get(self):
        pass


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    port = 8889
    app.listen(port)
    print('listening on localhost:{}'.format(port))
    tornado.ioloop.IOLoop.current().start()
