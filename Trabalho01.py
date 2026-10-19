"""
Trabalho 01 - Calculo Numerico
Ajuste de curva pelo Metodo dos Minimos Quadrados (MMQ)

Encontra phi(x) = a2*x^2 + a1*x + a0 que minimiza a soma dos
quadrados dos erros: S = sum( f(x_i) - phi(x_i) )^2

Uso:  python trabalho01.py pontos.csv
"""
# A string acima (docstring do modulo) documenta o objetivo do programa;
# ela nao e executada, serve apenas como descricao.

import csv  # modulo padrao do Python para ler/escrever arquivos CSV
import sys  # modulo padrao para acessar argumentos da linha de comando (sys.argv)


def ler_csv(caminho):  # define a funcao que recebe o caminho do arquivo CSV
    """Passo 1: le o arquivo CSV (uma linha 'x,f(x)' por ponto)."""
    xs, ys = [], []  # cria duas listas vazias: xs guarda os x, ys guarda os f(x)
    with open(caminho, newline="") as arq:  # abre o arquivo; o 'with' fecha automaticamente ao final
        for linha in csv.reader(arq):  # percorre o arquivo; cada 'linha' vira uma lista de strings, ex.: ['1.0', '2.5']
            if not linha or not linha[0].strip():  # se a linha esta vazia ou o primeiro campo so tem espacos...
                continue  # ...ignora essa linha e passa para a proxima
            xs.append(float(linha[0]))  # converte a 1a coluna (x) para numero real e adiciona em xs
            ys.append(float(linha[1]))  # converte a 2a coluna (f(x)) para numero real e adiciona em ys
    if len(xs) < 4:  # verifica se foram lidos menos de 4 pontos (exigencia do trabalho)
        raise ValueError("A entrada precisa ter pelo menos 4 pontos.")  # interrompe o programa com mensagem de erro
    return xs, ys  # devolve as duas listas de coordenadas


def montar_sistema_normal(xs, ys):  # define a funcao que monta as equacoes normais do MMQ
    """Passo 2: monta o sistema normal A*alpha = b.

    Com g0(x)=1, g1(x)=x, g2(x)=x^2, os elementos sao
    A[i][j] = sum x^(i+j)   e   b[i] = sum f(x) * x^i,
    com a incognita na ordem [a0, a1, a2].
    """
    # Monta a matriz A (3x3): para cada linha i e coluna j, soma x^(i+j) em todos os pontos.
    # Resultado: [[n, sum x, sum x^2], [sum x, sum x^2, sum x^3], [sum x^2, sum x^3, sum x^4]]
    A = [[sum(x ** (i + j) for x in xs) for j in range(3)] for i in range(3)]
    # Monta o vetor b (3 elementos): b[i] = soma de f(x) * x^i.
    # Resultado: [sum f(x), sum f(x)*x, sum f(x)*x^2]; zip(xs, ys) junta cada x com seu f(x)
    b = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(3)]
    return A, b  # devolve a matriz dos coeficientes e o vetor dos termos independentes


def eliminacao_gauss(A, b):  # define a funcao que resolve o sistema linear A*x = b
    """Passo 3: resolve A*alpha = b por eliminacao de Gauss com
    pivoteamento parcial, seguida de retrosubstituicao."""
    n = len(b)  # n = numero de equacoes/incognitas (aqui, 3)
    # Cria a matriz aumentada [A | b]: copia cada linha de A (linha[:]) e acrescenta b[i] no final.
    # A copia evita alterar a matriz A original.
    M = [linha[:] + [b[i]] for i, linha in enumerate(A)]  # matriz aumentada

    for k in range(n - 1):  # percorre as colunas 0 .. n-2; k e a coluna/linha do pivo atual
        # pivoteamento parcial: maior |elemento| da coluna k
        p = max(range(k, n), key=lambda i: abs(M[i][k]))  # p = indice da linha (de k ate n-1) com maior |M[i][k]|
        if abs(M[p][k]) < 1e-14:  # se ate o maior pivo e praticamente zero...
            raise ValueError("Sistema singular.")  # ...o sistema nao tem solucao unica; gera erro
        M[k], M[p] = M[p], M[k]  # troca a linha k com a linha p, colocando o maior pivo na diagonal
        for i in range(k + 1, n):  # para cada linha abaixo da linha do pivo...
            m = M[i][k] / M[k][k]  # calcula o multiplicador m = elemento a zerar / pivo
            for j in range(k, n + 1):  # percorre as colunas da k ate a ultima (inclui a coluna de b)
                M[i][j] -= m * M[k][j]  # Linha_i = Linha_i - m * Linha_k (zera o elemento M[i][k])

    # retrosubstituicao
    x = [0.0] * n  # cria o vetor solucao com n zeros
    for i in range(n - 1, -1, -1):  # percorre as linhas de baixo para cima (n-1, n-2, ..., 0)
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))  # soma os termos das incognitas ja calculadas (a direita da diagonal)
        x[i] = (M[i][n] - s) / M[i][i]  # isola x[i]: (termo independente - soma) / coeficiente da diagonal
    return x  # devolve a solucao [a0, a1, a2]


def minimos_quadrados(xs, ys):  # funcao que junta os passos 2 e 3
    """Executa o MMQ e devolve (a2, a1, a0)."""
    A, b = montar_sistema_normal(xs, ys)  # monta o sistema normal a partir dos pontos
    a0, a1, a2 = eliminacao_gauss(A, b)  # resolve o sistema; a solucao vem na ordem [a0, a1, a2]
    return a2, a1, a0  # devolve os coeficientes na ordem a2, a1, a0


def main():  # funcao principal do programa
    # Usa o caminho passado na linha de comando (sys.argv[1]); se nenhum for passado, usa "pontos.csv"
    caminho = sys.argv[1] if len(sys.argv) > 1 else "pontos.csv"
    xs, ys = ler_csv(caminho)  # Passo 1: le os pontos do arquivo
    a2, a1, a0 = minimos_quadrados(xs, ys)  # Passos 2 e 3: calcula os coeficientes da parabola
    # Passo 4: imprime na ordem a2, a1, a0
    print(f"alpha2 = {a2:.6f}")  # imprime a2 com 6 casas decimais (f-string com formato .6f)
    print(f"alpha1 = {a1:.6f}")  # imprime a1 com 6 casas decimais
    print(f"alpha0 = {a0:.6f}")  # imprime a0 com 6 casas decimais


if __name__ == "__main__":  # verdadeiro apenas quando o arquivo e executado diretamente (nao quando importado)
    main()  # chama a funcao principal
