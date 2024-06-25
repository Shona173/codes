import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu

def load_shader_source(filename):
    with open(filename, 'r') as file:
        return file.read()

def compile_shader(source, shader_type):
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        raise RuntimeError(gl.glGetShaderInfoLog(shader).decode('utf-8'))
    return shader

def create_shader_program(vertex_source, fragment_source):
    vertex_shader = compile_shader(vertex_source, gl.GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER)
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)
    if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
        raise RuntimeError(gl.glGetProgramInfoLog(program).decode('utf-8'))
    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)
    return program

def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glUseProgram(shader_program)
    # ここで描画コードを追加します
    glut.glutSwapBuffers()

def main():
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH)
    glut.glutCreateWindow(b'OpenGL Window')
    glut.glutDisplayFunc(display)

    vertex_source = load_shader_source('vertex_shader.glsl')
    fragment_source = load_shader_source('fragment_shader.glsl')
    global shader_program
    shader_program = create_shader_program(vertex_source, fragment_source)

    glut.glutMainLoop()

if __name__ == '__main__':
    main()
