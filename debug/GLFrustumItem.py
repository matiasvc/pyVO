from OpenGL.GL import *
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.Qt import QtGui

__all__ = ['GLFrustumItem']


class GLFrustumItem(GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem>`

    Displays three lines indicating origin and orientation of local coordinate system.

    """

    def __init__(self, frustumColor=(1,1,1,1), size=None, antialias=True, glOptions='translucent'):
        GLGraphicsItem.__init__(self)
        if size is None:
            size = QtGui.QVector3D(1, 1, 1)
        self.antialias = antialias
        self.setSize(size=size)
        self.setGLOptions(glOptions)
        assert len(frustumColor) == 4, f'frustum color must be a 4 element tuple, it is: {frustumColor}'
        self.frustumColor = frustumColor

    def setSize(self, x=None, y=None, z=None, size=None):
        """
        Set the size of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
        if size is not None:
            x = size.x()
            y = size.y()
            z = size.z()
        self.__size = [x, y, z]
        self.update()

    def size(self):
        return self.__size[:]

    def paint(self):

        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glEnable( GL_BLEND )
        # glEnable( GL_ALPHA_TEST )
        self.setupGLState()

        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glBegin(GL_LINES)

        x, y, z = self.size()
        glColor4f(0, 0, 1, .6)  # z is blue
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, z)

        glColor4f(0, 1, 0, .6)  # y is green
        glVertex3f(0, 0, 0)
        glVertex3f(0, y, 0)

        glColor4f(1, 0, 0, .6)  # x is red
        glVertex3f(0, 0, 0)
        glVertex3f(x, 0, 0)

        fx = 525.0/640.0
        fy = 525.0/480.0

        p1 = (fy*z, fx*z, z)
        p2 = (-fy * z, fx * z, z)
        p3 = (-fy * z, -fx * z, z)
        p4 = (fy * z, -fx * z, z)

        glColor4f(*self.frustumColor)
        glVertex3f(0, 0, 0)
        glVertex3f(*p1)

        glVertex3f(0, 0, 0)
        glVertex3f(*p2)

        glVertex3f(0, 0, 0)
        glVertex3f(*p3)

        glVertex3f(0, 0, 0)
        glVertex3f(*p4)

        glVertex3f(*p1)
        glVertex3f(*p2)

        glVertex3f(*p2)
        glVertex3f(*p3)

        glVertex3f(*p3)
        glVertex3f(*p4)

        glVertex3f(*p4)
        glVertex3f(*p1)

        glEnd()
