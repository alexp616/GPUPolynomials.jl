function my_function(x, y=10)
    if isa(y, Nothing)
        println("Optional parameter not provided")
        # Do something when y is not provided
    else
        println("Optional parameter provided: $y")
        # Do something when y is provided
    end
    # Do something with x and y
end

my_function(5)    # Output: Optional parameter provided: 10
my_function(5, 20)    # Output: Optional parameter provided: 20