#pragma once

#include "scannertools_sql.pb.h"
#include <pqxx/pqxx>
#include <memory>

namespace scanner {
  std::unique_ptr<pqxx::connection> sql_connect(SQLConfig sql_config);
}
