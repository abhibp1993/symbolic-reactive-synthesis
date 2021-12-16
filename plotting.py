import pickle
import matplotlib.pyplot as plt


def compare_mem_const(DIM, sym_mem_const, enum_mem_const):
    plt.plot(DIM, sym_mem_const, marker = '.',  markersize = 10, label = 'sym_mem_const', linestyle = 'solid')
    plt.plot(DIM, enum_mem_const, marker = '*',  markersize = 10, label = 'enum_mem_const', linestyle = 'solid')
    plt.title('Construction Memory Comparison')
    plt.xlabel('Gridworld Dimension')
    plt.ylabel('Construction Memory (bytes)')
    plt.legend()
    plt.show()
    
def compare_mem_solve(DIM, sym_mem_solve, enum_mem_solve):
    plt.plot(DIM, sym_mem_solve, marker = '.',  markersize = 10, label = 'sym_mem_solve', linestyle = 'solid')
    plt.plot(DIM, enum_mem_solve, marker = '*',  markersize = 10, label = 'enum_mem_solve', linestyle = 'solid')
    plt.title('Solving Memory Comparison')
    plt.xlabel('Gridworld Dimension')
    plt.ylabel('Solving Memory (bytes)')
    plt.legend()
    plt.show()
    
def compare_mem_total(DIM, sym_mem_total, enum_mem_total):
    plt.plot(DIM, sym_mem_total, marker = '.',  markersize = 10, label = 'sym_mem_total', linestyle = 'solid')
    plt.plot(DIM, enum_mem_total, marker = '*',  markersize = 10, label = 'enum_mem_total', linestyle = 'solid')
    plt.title('Total Memory Comparison')
    plt.xlabel('Gridworld Dimension')
    plt.ylabel('Total Memory (bytes)')
    plt.legend()
    plt.show()
    
def compare_time_const(DIM, sym_time_const, enum_time_const):
    plt.plot(DIM, sym_time_const, marker = '.',  markersize = 10, label = 'sym_time_const', linestyle = 'solid')
    plt.plot(DIM, enum_time_const, marker = '*',  markersize = 10, label = 'enum_time_const', linestyle = 'solid')
    plt.title('Construction Time Comparison')
    plt.xlabel('Gridworld Dimension')
    plt.ylabel('Construction Time (ms)')
    plt.legend()
    plt.show()
    
def compare_time_solve(DIM, sym_time_solve, enum_time_solve):
    plt.plot(DIM, sym_time_solve, marker = '.',  markersize = 10, label = 'sym_time_solve', linestyle = 'solid')
    plt.plot(DIM, enum_time_solve, marker = '*',  markersize = 10, label = 'enum_time_solve', linestyle = 'solid')
    plt.title('Solving time Comparison')
    plt.xlabel('Gridworld Dimension')
    plt.ylabel('Solving Time (ms)')
    plt.legend()
    plt.show()
    
def compare_time_total(DIM, sym_time_total, enum_time_total):
    plt.plot(DIM, sym_time_total, marker = '.',  markersize = 10, label = 'sym_time_total', linestyle = 'solid')
    plt.plot(DIM, enum_time_total, marker = '*',  markersize = 10, label = 'enum_time_total', linestyle = 'solid')
    plt.title('Total Time Comparison')
    plt.xlabel('Gridworld Dimension')
    plt.ylabel('Total Time (bytes)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    with open("benchmark.pkl", "rb") as file:
        data = pickle.load(file)
    print(data)

    DIM = data['DIM']
    sym_mem_const = data['sym_mem_const']
    sym_mem_solve = data['sym_mem_solve']
    sym_mem_total = data['sym_mem_total']
    sym_time_const = data['sym_time_const']
    sym_time_solve = data['sym_time_solve']
    sym_time_total = data['sym_time_total']
    
    enum_mem_const = data['enum_mem_const']
    enum_mem_solve = data['enum_mem_solve']
    enum_mem_total = data['enum_mem_total']
    enum_time_const = data['enum_time_const']
    enum_time_solve = data['enum_time_solve']
    enum_time_total = data['enum_time_total']
    
    compare_mem_const(DIM, sym_mem_const, enum_mem_const)
    
    compare_mem_solve(DIM, sym_mem_solve, enum_mem_solve)
    
    compare_mem_total(DIM, sym_mem_total, enum_mem_total)
    
    compare_time_const(DIM, sym_time_const, enum_time_const)
    
    compare_time_solve(DIM, sym_time_solve, enum_time_solve)
    
    compare_time_total(DIM, sym_time_total, enum_time_total)


