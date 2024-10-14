import threading


class CamThread(threading.Thread):

    def __init__(self, preview_name, cam_id, callback_f):
        """ Webcam thread instance

        Args:
            preview_name: displayed log message name
            cam_id: webcam id
            callback_f: function to execute when thread is started
        """
        threading.Thread.__init__(self)
        self.previewName = preview_name
        self.camID = cam_id
        self.callback_f = callback_f

    def run(self):
        print("Starting " + self.previewName)
        self.callback_f(self.previewName, self.camID)
