def solve_tridiagonal(a, b, c, d):
    """
    Resuelve sistema tridiagonal Ax = d usando el algoritmo de Thomas
    donde la matriz A tiene:
    - a: diagonal inferior (subdiagonal)
    - b: diagonal principal  
    - c: diagonal superior (superdiagonal)
    - d: vector independiente (lado derecho)
    
    Retorna: vector solución x
    """
    n = len(d)
    
    # Crear copias para evitar modificar los originales
    b_mod = b[:]
    d_mod = d[:]
    
    # Forward elimination (eliminación hacia adelante)
    for i in range(1, n):
        factor = a[i-1] / b_mod[i-1]
        b_mod[i] -= factor * c[i-1]
        d_mod[i] -= factor * d_mod[i-1]
    
    # Back substitution (sustitución hacia atrás)
    x = [0.0] * n
    x[n-1] = d_mod[n-1] / b_mod[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = (d_mod[i] - c[i] * x[i+1]) / b_mod[i]
    
    return x


def metodo_implicito_calor(alpha, f, dx, dt, T):
    """
    Método implícito para resolver la ecuación del calor sin usar numpy
    """
    # Número de puntos espaciales (incluyendo fronteras)
    N = int(1/dx) + 1
    # Número de pasos temporales
    M = int(T/dt) + 1
    
    # Inicializar matriz solución usando listas de listas
    U = [[0.0 for _ in range(N)] for _ in range(M)]
    
    # Vector de posiciones espaciales
    x = [i * dx for i in range(N)]
    
    # Aplicar condición inicial
    for j in range(N):
        U[0][j] = f(x[j])
    
    # Condiciones de frontera (Dirichlet homogéneas)
    for n in range(M):
        U[n][0] = 0.0   # u(0,t) = 0
        U[n][-1] = 0.0  # u(1,t) = 0
    
    # Parámetro de estabilidad
    r = alpha * dt / (dx**2)
    
    # Para el sistema tridiagonal A * U_next = b
    # La matriz A es (N-2) x (N-2) para los puntos interiores
    size = N - 2
    
    # Diagonales del sistema tridiagonal
    a = [-r] * (size - 1)      # diagonal inferior
    b = [1 + 2*r] * size       # diagonal principal  
    c = [-r] * (size - 1)      # diagonal superior
    
    # Iteración temporal
    for n in range(M-1):
        # Vector lado derecho (nodos internos solamente)
        rhs = [U[n][j] for j in range(1, N-1)]
        
        # Resolver sistema tridiagonal: A * U_next = rhs
        # Reemplaza la línea: U[n+1, 1:-1] = np.linalg.solve(A, b)
        solution = solve_tridiagonal(a, b, c, rhs)
        
        # Asignar la solución a los puntos internos
        for j in range(size):
            U[n+1][j+1] = solution[j]
    
    return U, x


# Función de ejemplo para probar
def f_triangular(x):
    """Función triangular: 1 - |2x - 1|"""
    return 1 - abs(2 * x - 1)


# Parámetros de prueba
if __name__ == "__main__":
    alpha = 1.0
    dt = 0.001
    dx = 0.05
    T = 0.1
    
    print("Ejecutando método implícito sin numpy...")
    U, x_grid = metodo_implicito_calor(alpha, f_triangular, dx, dt, T)
    
    print(f"Dimensiones de la matriz solución: {len(U)} x {len(U[0])}")
    print(f"Número de puntos espaciales: {len(x_grid)}")
    print(f"Número de pasos temporales: {len(U)}")
    
    # Mostrar algunos valores
    r = alpha * dt / (dx**2)
    print(f"\nParámetro r = {r:.4f}")
    
    print(f"\nCondición inicial (t=0):")
    print(f"u(0.25, 0) = {U[0][int(0.25/dx)]:.4f}")
    print(f"u(0.50, 0) = {U[0][int(0.50/dx)]:.4f}")
    print(f"u(0.75, 0) = {U[0][int(0.75/dx)]:.4f}")
    
    print(f"\nSolución en t = T = {T}:")
    print(f"u(0.25, {T}) = {U[-1][int(0.25/dx)]:.4f}")
    print(f"u(0.50, {T}) = {U[-1][int(0.50/dx)]:.4f}")
    print(f"u(0.75, {T}) = {U[-1][int(0.75/dx)]:.4f}")
