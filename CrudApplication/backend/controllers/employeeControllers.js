const Employee = require("../model/Employee");

exports.createEmployee = async (req, res) => {
    try {
        const employee = new Employee(req.body);
        await employee.save();
        res.json(employee);

    } catch (error) {
        res.json(error);
    }

};

exports.findAllEmployee = async (req, res) => {
    try {
        const employee = await Employee.find();
        res.json(employee);

    } catch (error) {
        res.status(500).json({
            message: "Server Error",
            error: error.message
        });
    }
};

exports.findEmployeeById = async (req, res) => {
    try {
        const employee = await Employee.findbyId(req.params.id);
        res.json(employee);
    } catch (error) {
        res.status(500).json({
            message: "Server Error/Cant Find",
            error: error.message
        });
    }
};

exports.updateEmployee = async (req, res) => {
    try {
        const employee = await Employee.findByIdAndUpdate(
            req.params.id,
            req.body,
            { new: true }
        )
    } catch (error) {
        res.status(500).json({
            message: "Update Failed",
            error: error.message
        });
    }
};

exports.deleteEmployee = async (req,res)=>{

    await Employee.findByIdAndDelete(req.params.id);

    res.json("Employee Deleted");

};