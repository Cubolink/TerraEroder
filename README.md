# Tarea 3 - CC7515 - Implementación de Curvas de Nivel de un Terreno
**Computación en GPU**

**Estudiante: Joaquín Cruz Cancino**

## Instrucciones de compilación

### Build Cmake Project
Primero se deben construir con CMake un proyecto compilable.
Para ello, en una terminal corra el siguiente comando para crear el proyecto dentro de la carpeta `build`.

```
mkdir build
cmake -S . -B build
```

Es posible que tenga que especificar un generador.

* Por ejemplo, si desea compilar con MinGW, deberá añadir la flag `-G "MinGW Makefiles`.
* O si desea utilizar VS 2017, para una arquitectura de 64 bits, `-G "Visual Studio 15 2017 Win64"`

Refiérase a `cmake --help` para ver los generadores disponibles, asegúrese de utilizar uno compatible con el compilador
que desea usar.

### Compilar los build sources y correr el programa

Una vez generados los archivos con cmake, puede ir a la carpeta build (`cd build`), y compilar. La forma de hacerlo puede
variar dependiendo del compilador. A continuación se proveen algunos casos comunes.

**Unix-like (g++)**
```
make
```
El ejecutable debiera estar en la carpeta `build/src` como `terra_eroder.exe`.

**MinGw**
```
mingw32-make
```
Luego, el ejecutable se encontrará en la carpeta `build/src` como `terra_eroder.exe`.

**Visual Studio desde la terminal**
```
msbuild terraeroder.sln
```
Luego, el ejecutable se encontrará en la carpeta `build/src/Debug` como `terra_eroder.exe`. Sin embargo, probablemente no funcione.
En este caso, deberá mover el ejecutable a `build/src`, y ahí podrá ejecutarlo.

También puede ejecutar `msbuild terraeroder.sln` sin ninguna flag. En ese caso, el ejecutable se encontrará en `build/src`, pero
deberá manualmente copiar la carpeta `src`

**Visual Studio desde el IDE**

En este caso, deberá abrir `terraeroder.sln` con su Visual Studio y compilar y ejecutar desde allí el proyecto `terra_eroder`.

## Instrucciones de Ejecución

Al correr el ejecutable se encontrará con una cámara en primera persona, y con una ventana de ImGui.

*Controles de la cámara*
```
WASD -> Moverse hacia adelante/atrás y hacia los lados
Space -> Moverse hacia arriba
L-Shift -> Moverse hacia abajo

Flechas -> Mirar alrededor
```

**Cambiar posición de la luz**

Mediante la ventana de ImGui, puedes deslizar la posición de la luz. Se proveen deslizadores para las tres coordenadas.

**Isolineas**

Presionando la tecla `TAB`, o bien desde el selector de shaders en la ventana de ImGui, puede elegir cómo mostrar el terreno.
Con esto, podrá alternar entre ver el terreno colorizado por altura, ver las isolineas sobre el terreno en gris, o ver la triangulación.

Además, con ImGui puede configurar cuántas isolineas tener y a qué alturas.

**Modificar el terreno**

Si presiona la tecla `T`, o bien presiona el botón `Randomize` en ImGui, se cambiarán aleatoriamente las posiciones de
los vértices desde el vertex shader. Puede presionar la tecla `R` para restablecer al terreno original.

## Estructura del proyecto

La carpeta `Dependencies` contiene librerías necesarias para todo el proyecto. La carpeta `src/vendor` también contiene
dependencias externas, pero que funcionan más como subproyectos externos.

La carpeta `resources` contiene las shaders que se utilizan durante la ejecución de la aplicación. 

La carpeta `src/renderer` contiene abstracciones para renderizar con OpenGL.

El resto de los archivos en `src` forman el programa principal. Este proyecto se transformará en Terraeroder, el proyecto
final del curso para erosionar terreno, y por eso el nombre. Para fines de la tarea 3 se adaptaron algunas cosas, pero quedó el nombre.