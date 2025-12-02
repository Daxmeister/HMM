import sys


class HMM_model():
    def __init__(self):
        pass
        
    def run_task_0(self):
        A,B,pi = self.initialize_parameters(0)
        state_estimate = self.multiplication_vector_matrix(pi, A)
        observation_probabilities = self.multiplication_vector_matrix(state_estimate, B)
        self.output_vector(observation_probabilities)
        
    
    def run_task_1(self):
        A,B,pi, emissions = self.initialize_parameters(1)
        value = self.forward_algorithm(A, B, pi, emissions)
        print(value)
        
    def initialize_parameters(self, task_number=0):
        A = self.matrix_from_input(sys.stdin.readline())
        B = self.matrix_from_input(sys.stdin.readline())
        pi = self.matrix_from_input(sys.stdin.readline())
        pi = pi[0] #Since it only has one object
        if task_number == 1:
            emissions = self.matrix_from_input(sys.stdin.readline(), True)
            emissions = emissions[0] #Only vector
            return A, B, pi, emissions
        else:
            return A, B, pi
    
    
    def matrix_from_input(self, line, is_emission=False):
        line = line.split()
        if is_emission: # Since emission vector has it's own input format...........
            rows = 1
            columns = int(line[0])
            matrix_values = list(map(int, line[1:])) # emission has integers, not floats
        else:
            rows = int(line[0])
            columns = int(line[1])
            matrix_values = list(map(float, line[2:]))
        matrix = []
        
        for row in range(rows):
            first = row * columns
            last = first + columns
            matrix.append(matrix_values[first:last])         
        return matrix

    

    def multiplication_vector_matrix(self, vector, matrix):
        # Mutiplies a 1xn row vector with a nxm matrix
        # Used in task 0
        n = len(vector)
        m = len(matrix[0])
        output_vector = [] 
        for j in range(m):
            total = 0.0
            for i in range(n):
                total += vector[i] * matrix[i][j]
            output_vector.append(total)
        #Outputs a 1xm vector
        return output_vector
    
    def element_wise_product(self, a, b):
        # Element wise product of two 1xn vectors (lists)
        c = []
        for index in range(len(a)):
            c.append(a[index]*b[index])
        return c
    
    def column_from_B(self, B, observation):
        # Outputs a the column of state probabilities given observation
        # Note that observation must be the "index"
        output_vector = []
        for row_number in range(len(B)):
            output_vector.append(B[row_number][observation])
        return output_vector
    
    def calculate_state_probabilities(self, A,previous_alpha):
        # Calculates probability of being in certain state 
        # Used in task 1
        
        output_vector = []
        for row_index in range(len(A)):
            row_sum = 0
            for column_index in range(len(A[0])):
                row_sum += A[row_index][column_index]*previous_alpha[column_index]
            output_vector.append(row_sum)
        return output_vector
    
    def forward_algorithm(self, A,B,pi,emissions):
        alpha = []
        
        #Step 1
        B_column = self.column_from_B(B, emissions[0])
        alpha = self.element_wise_product(pi, B_column)
        
        # Step 2
        for emission_number in range(1, len(emissions)):
            B_column = self.column_from_B(B, emissions[emission_number])
            #state_probabilities = self.calculate_state_probabilities(A, alpha)
            #alpha = self.element_wise_product(state_probabilities, B_column)
            state_probabilities = self.multiplication_vector_matrix(alpha, A)
            alpha = self.element_wise_product(state_probabilities, B_column)
        
        # Step 3
        total = 0
        for value in alpha:
            total += value
        return total
        
        
        
    
    
    
    def output_vector(self, vector):
        string = "1 "
        string += str(len(vector))
        string += " "
        for i in range(len(vector)):
            string += str(vector[i])
            string += " "
        print(string)
    
    
        
        

object = HMM_model()
object.run_task_1()
#object.run_task_0()

