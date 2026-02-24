const express = require("express");

const router = express.Router();

const {
createEmployee,
findAllEmployee,
findEmployeeById,
updateEmployee,
deleteEmployee

} = require("../controllers/employeeControllers");


router.post("/employees",createEmployee);

router.get("/employees",findAllEmployee);

router.get("/employees/:id",findEmployeeById);

router.put("/employees/:id",updateEmployee);

router.delete("/employees/:id",deleteEmployee);


module.exports = router;