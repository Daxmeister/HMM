import sys
import math

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
        T = len(emissions) # Number of time steps

        alpha = self.forward_algorithm(A, B, pi, emissions, T)
         # Step 3 - Final step
        print(sum(alpha[-1]))
        
    def run_task_2(self):
        A,B,pi, emissions = self.initialize_parameters(2)
        probable_emissions = self.viterbi_algorithm(A, B, pi, emissions)
        self.output_nonvector(probable_emissions)
        
    def run_task_3(self):
        # Initialize lambda
        A,B,pi, emissions = self.initialize_parameters(2)
        N = len(A) # Number of states
        M = len(B[0]) # Number of different emissions
        T = len(emissions) # Number of time steps
        
       
        
        # Re-estimate lambda
        improved = True
        while improved:
             # Compute alpha, beta, di-gamma and gamma
            alpha, scale, log_probability = self.forward_algorithm_scaled( A,B,pi,emissions, T)
            beta = self.backwards_algorithm_scaled(A,B,emissions, N, T, scale)
            di_gamma = self.di_gamma_function(A,B,emissions, alpha, beta, N, T)
            gamma = self.gamma_function( di_gamma, N, T)
            
            # Re-estimate lambda
            new_A = self.approximate_A(di_gamma, gamma, N, T)
            new_B = self.approximate_B(emissions, gamma, N, T, M)
            new_pi = self.approximate_pi(gamma, N)
            
            # Evaluate improvement
            old_model_probability = sum(self.forward_algorithm(A, B, pi, emissions, T)[-1])
            new_model_probability = sum(self.forward_algorithm(new_A, new_B, new_pi, emissions, T)[-1])
            print(old_model_probability)
            print(new_model_probability)
            if new_model_probability <= old_model_probability:
                improved = False
            else:
                A = new_A
                B = new_B
                pi = new_pi
            
    
        
        # Evaluate improvement
        
    def di_gamma_function(self, A,B,emissions, alpha, beta, N, T):
        
        # final_alpha_values = sum(alpha[-1]) # This CAN NOT be zero, fix for large input. OLD
        
        # di_gamma is (T-1)*N*N
        di_gamma  = [
                        [
                                [0.0]*N 
                            for _ in range(N)
                        ]
                    for _ in range(T-1)
                    ]
        # After scaling, we need to recalculate the denominator at every step
            
        
        for t in range(0, T-1): # Thre can be no transition from the last time instance, thus T-1
            emission_t1 = emissions[t+1]
            denominator_t = 0.0
            # Denominator is recalculated for every t since scaling "accumulates"
            for i in range(N):
                for j in range(N):
                    denominator_t += alpha[t][i] * A[i][j] * B[j][emission_t1] * beta[t+1][j]
                    
            
            for i in range(N):
                for j in range(N):
                    di_gamma[t][i][j] = alpha[t][i] *A[i][j] * B[j][emissions[t+1]]*beta[t+1][j] / denominator_t
                    
        return di_gamma
    
    def approximate_A(self, di_gamma, gamma, N, T):

        A = [[0.0]*N for _ in range(N)]
        
        for i in range(N):
            for j in range(N):
                nominator = 0.0
                denominator = 0.0
                for t in range(T-1):
                    nominator += di_gamma[t][i][j]
                    denominator += gamma[t][i]    
                if denominator != 0: # In case we believe that we never visit state i, denominator is 0.
                    A[i][j] = nominator/denominator
                else:
                    A[i][j] = 0.0
        
        return A
    
    def approximate_B(self, emissions, gamma, N, T, M):
        
        B = [[0.0]*M for _ in range(N)]
        
        for j in range(N):
            for k in range(M):
                nominator = 0.0
                denominator = 0.0
                for t in range(T-1):
                    if emissions[t]==k: # Implements the indicator function
                        nominator += gamma[t][j]
                    denominator += gamma[t][j]
                if denominator != 0:
                    B[j][k] = nominator/denominator
                else:
                    B[j][k] = 0.0
        return B
    
    def approximate_pi(self, gamma, N):
        pi = []
        
        for i in range(N):
            pi.append(gamma[0][i])    
        
        return pi    
                    
                
                
        
        
    
    def gamma_function(self, di_gamma, N, T):
        
        gamma = [[0.0]*N for _ in range(T-1)]
        
        for t in range(T-1): # Goes up till T-2 since di_gamma is only defined for t=[0,T-2]
            for i in range(N):
                gamma[t][i]= sum(di_gamma[t][i]) #Summarizes over all "j"
                
        # No gamma for all T, only up to T-1 (or T-2 using python index). Fix?
        return gamma
                                
    def initialize_parameters(self, task_number=0):
        A = self.matrix_from_input(sys.stdin.readline())
        B = self.matrix_from_input(sys.stdin.readline())
        pi = self.matrix_from_input(sys.stdin.readline())
        pi = pi[0] #Since it only has one object
        if task_number == 1 or task_number == 2:
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

    
    def forward_algorithm(self, A,B,pi,emissions, T):
        # Saves and returns entire alpha matrix. Implemented for task 3
        alpha = []
        
        #Step 1 - Initialize
        B_column = self.column_from_B(B, emissions[0])
        alpha.append(self.element_wise_product(pi, B_column))
        
        # Step 2 - Middle part
        for t in range(1, T):
            B_column = self.column_from_B(B, emissions[t])
            state_probabilities = self.multiplication_vector_matrix(alpha[t-1], A)
            alpha.append(self.element_wise_product(state_probabilities, B_column))
        
        return alpha
    
    def forward_algorithm_scaled(self, A,B,pi,emissions, T):
        # Saves and returns entire alpha matrix. Implemented for task 3
        alpha = []
        scale = [0.0]*T
        
        #Step 1 - Initialize at t=0
        
        # Calculate alpha at t=0
        B_column = self.column_from_B(B, emissions[0])
        alpha.append(self.element_wise_product(pi, B_column))
        
        # Scale alpha and save scaleing factor
        # 1.0/sum is choosen as scaling factor so that all for time t the alpha vector holds a probability distribution for all states
        scale[0] = 1.0 / sum(alpha[0]) # Calculate scaling factor. Scaling this way means that
        alpha[0] = [value * scale[0] for value in alpha[0]] # Multiply every alpha by scaling factor
        
        
        # Step 2 - Middle part
        for t in range(1, T):
            B_column = self.column_from_B(B, emissions[t])
            state_probabilities = self.multiplication_vector_matrix(alpha[t-1], A)
            alpha.append(self.element_wise_product(state_probabilities, B_column))
            scale[t] = 1.0 / sum(alpha[t]) # Calculate scaling factor
            alpha[t] = [value * scale[t] for value in alpha[t]] # Multiply every alpha by scaling factor
        
        # log probability - needed for baum-welch
        log_probability = 0
        for t in range(T):
            log_probability += math.log(1 / scale[t]) # The math is important here - this is the log version of what problem 1 demands
        
        return alpha, scale, log_probability
    
    def backwards_algorithm_scaled(self, A,B,emissions, N, T, scale):
        
        beta = [[0.0]*N for _ in range(T)]
        scale = [0.0]*T


        # Step 1 initialize
        beta[T-1] = [1]*N
        
        
        # Step 2 - middle part
        for t in range(T-2, -1, -1): # From T to 0 (but indexes start with 0, go to T-1)
            for i in range(N): # For every state
                total = 0
                for j in range(N):
                    total += beta[t+1][j] * A[i][j] * B[j][emissions[t+1]]
                beta[t][i] = total * scale[t+1] # We use tha same scaling as we used for alpha for consistency

        return beta
    
    
    def backwards_algorithm(self, A,B,emissions, N, T):
        
        beta = [[0.0]*N for _ in range(T)]

        # Step 1 initialize
        beta[T-1] = [1]*N
        
        # Step 2 - middle part
        for t in range(T-2, -1, -1): # From T to 0 (but indexes start with 0, go to T-1)
            for i in range(N): # For every state
                total = 0
                for j in range(N):
                    total += beta[t+1][j] * A[i][j] * B[j][emissions[t+1]]
                beta[t][i] = total

        return beta
        
        

    
    def viterbi_algorithm(self, A, B, pi, emissions):
        delta = []
        previous_states = []
        n_states = len(A)
        
        # Step 1 - initialize
        
        emission_number = emissions[0]
        B_column = self.column_from_B(B, emission_number)
        delta.append(self.element_wise_product(pi, B_column))
        previous_states.append([None]*n_states)
        
        # Step 2 - Iterate
        
        # For every emission after the first, calculate delta
        for n in range(1, len(emissions)): 
            delta_n, previous_state_n = self.viterbi_step_2(A, B, emissions, n, n_states, delta)
            delta.append(delta_n)
            previous_states.append(previous_state_n)

        
        # Step 3 - Terminate
        
        probable_states = []
        
        # Calculate most likely state at end
        i = len(emissions)-1 #
        most_probable_state_at_i_minus_1 = delta[i].index(max(delta[i])) 
        probable_states.append(most_probable_state_at_i_minus_1)
        
        # Calculate all preceding states
        for i in range(len(emissions)-1, 0, -1): 
            most_probable_state = most_probable_state_at_i_minus_1 # This line exists for pedagogical purposes
            most_probable_state_at_i_minus_1 = previous_states[i][most_probable_state] # Identifies most likely state at time i-1
            probable_states.append(most_probable_state_at_i_minus_1) 
        
        # Reverse order 
        probable_states.reverse()
        return probable_states
                    
        
    
    def viterbi_step_2(self, A, B, emissions, n, n_states, delta):
        emission = emissions[n]
        delta_n=[]
        most_likely_previous_states_n=[]
        for state_i in range(n_states):
            max_probability = 0 
            most_likely_previous_state = None
            for state_j in range(n_states):
                value = delta[n-1][state_j] * A[state_j][state_i] * B[state_i][emission]
                if value > max_probability:
                    max_probability = value
                    most_likely_previous_state = state_j
            delta_n.append(max_probability)
            most_likely_previous_states_n.append(most_likely_previous_state)
        return delta_n, most_likely_previous_states_n
    
            
    
    def output_vector(self, vector):
        string = "1 "
        string += str(len(vector))
        string += " "
        for i in range(len(vector)):
            string += str(vector[i])
            string += " "
        print(string)
    
    def output_nonvector(self, vector):
        out_string = ""
        for element in vector:
            out_string = out_string + " " + str(element)
        print(out_string)
    
    
        
        

object = HMM_model()
object.run_task_3()
#object.run_task_0()

