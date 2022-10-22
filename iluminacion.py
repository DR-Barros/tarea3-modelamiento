from random import randint, random
import glfw
from OpenGL.GL import *
import numpy as np
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
from grafica.assets_path import getAssetPath

class Controller:
    def __init__(self):
        self.theta = np.pi/2
        self.phi = 0
        self.position = np.array([5*np.sin(self.theta)*np.cos(self.phi), 5*np.sin(self.theta)*np.sin(self.phi), 5*np.cos(self.theta)])
    def arriba(self):
        self.theta -= np.pi/40
        self.__actualizarposition__()
    def abajo(self):
        self.theta += np.pi/40
        self.__actualizarposition__()
    def derecha(self):
        self.phi += np.pi/40
        self.__actualizarposition__()
    def izquierda(self):
        self.phi -= np.pi/40
        self.__actualizarposition__()
    def __actualizarposition__(self):
        self.position = np.array([5*np.sin(self.theta)*np.cos(self.phi), 5*np.sin(self.theta)*np.sin(self.phi), 5*np.cos(self.theta)])

controller = Controller()

# funcion para facilitar inicialización
def createGPUShape(shape, pipeline):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)
    return gpuShape
def esferaPhong(N):
    vertices = []
    faces = []
    angulo = 2 * np.pi
    n = int(N/2)
    for i in range(N+1):
        omega = i/N * angulo
        if i == N:
            omega = 0
            print(0)
        for j in range(n+1):
            theta = j/(N)* angulo
            vertices += [
                [np.sin(theta)*np.cos(omega), np.sin(theta)*np.sin(omega), np.cos(theta),
                i/N, j/n,
                np.sin(theta)*np.cos(omega), np.sin(theta)*np.sin(omega), np.cos(theta)],
            ]
            if i != N:
                faces +=[
                    [(n+1)*i+j, (n+1)*i+1+j, ((n+1)*(i+1)+j)%((N+1)*(n+1))],
                    [(n+1)*i+1+j, ((n+1)*(i+1)+j)%((N+1)*(n+1)), ((n+1)*(i+1)+1+j)%((N+1)*(n+1))]
                ]
    indices = []
    vertexData = []
    index = 0
    for face in faces:
        vertex = vertices[face[0]]
        vertexData += vertex
        vertex = vertices[face[1]]
        vertexData += vertex
        vertex = vertices[face[2]]
        vertexData += vertex

        indices += [index, index + 1, index + 2]
        index += 3  
    
    return bs.Shape(vertexData, indices)


def on_key(window, key, scancode, action, mods):
    #Si no se presiona una tecla no hace nada
    if action != glfw.PRESS:
        return

    global controller
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    if key == glfw.KEY_W:
        controller.arriba()
    if key == glfw.KEY_S:
        controller.abajo()
    if key == glfw.KEY_D:
        controller.derecha()
    if key == glfw.KEY_A:
        controller.izquierda()


def main():

    #inicializa glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)
    
    width = 1500
    height = 1000

    window = glfw.create_window(width, height, "Tarea 2", None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)
    
    glfw.make_context_current(window)

    #Conecta la funcion callback 'on_key' para manejar los eventos del teclado
    glfw.set_key_callback(window, on_key)

    #Ensamblando el shader program
    pipeline = ls.SimpleTexturePhongShaderProgram()


    #mandar a OpenGL a usar el shader program
    glUseProgram(pipeline.shaderProgram)

    #setiando el color de fondo
    glClearColor(0.05, 0.05, 0.15, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)


    global controller
    #Proyeccion
    proyeccion = tr.perspective(60, float(1500)/float(1000), 0.001, 200)
    view = tr.lookAt(
        controller.position,
        np.array ([0,0, 0]),
        np.array([0,0,1])
    )

    #Crear shapes en la GPU memory
    planeta = createGPUShape(esferaPhong(100), pipeline)
    planeta.texture = es.textureSimpleSetup(
        getAssetPath("jupiter.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR
    )

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 0.7, 0.7, 0.7)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), 1.0, 0.7, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 0, 0, 0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "lightPosition"), 1, 1, 1)
    
    glUniform1ui(glGetUniformLocation(pipeline.shaderProgram, "shininess"), 100)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "constantAttenuation"), 0.001)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "quadraticAttenuation"), 0.01)

    while not glfw.window_should_close(window):
        #usar GLFW para chequear input events
        glfw.poll_events()

        view = tr.lookAt(
            controller.position,
            np.array ([0,0, 0]),
            np.array([0,0,1])
        )
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, proyeccion)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)


        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        pipeline.drawCall(planeta)


        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)



    # freeing GPU memory
    planeta.clear()

    glfw.terminate()

if __name__ == "__main__":
    main()
